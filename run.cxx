#include "include/cpd.hpp"

char *getCmdOption(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

vector<int> getVectorCmdOption(char **begin, char **end,
                               const std::string &option) {
  vector<int> ret_vector = {};
  char **itr = std::find(begin, end, option);
  ++itr;
  while (itr != end && isdigit((*itr)[0])) {
    ret_vector.push_back(atoi(*itr));
    ++itr;
  }
  return ret_vector;
}

int main(int argc, char **argv) {
  int rank, np; //, n, pass;
  int const in_num = argc;
  char **input_str = argv;

  char *tensor; // which tensor    c / r / r2 / o /
  int method;   // 0 simple 1 Local-simple 2 DT 3 Local-DT 4 PP 5 Local-PP
  int ppmethod;   // 0 simple pp 1 new pp
  int seed = 1;
  bool use_msdt = false;
  bool renew_ppoperator = false;
  double update_percentage_pp; // pp update ratio. For each sweep only update
                               // update_percentage_pp*N matrices.
  /*
  p : poisson operator
  p2 : poisson operator with doubled dimension (decomposition is not accurate)
  c : decomposition of designed tensor with constrained collinearity
  r : decomposition of tensor made by random matrices
  r2 : random tensor
  o1 : coil-100 dataset
  */
  int dim;                         // number of dimensions
  vector<int> sizes;               // tensor size in each dimension
  vector<int> processor_mesh = {}; // the physical processor mesh grid
  int R;                           // decomposition rank
  int issparse;                    // whether use the sparse routine or not
  double tol;                      // global convergance tolerance
  double pp_res_tol;               // pp restart tolerance
  double lambda_;                  // regularization param
  double magni;                    // pp update magnitude
  char *filename;                  // output csv filename
  double col_min;                  // collinearity min
  double col_max;                  // collinearity max
  double ratio_noise;              // collinearity ratio of noise
  double timelimit = 5e7;          // time limits
  int maxsweep = 5e7;              // maximum sweeps
  int resprint = 1;
  char *tensorfile;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  MPI_File fh;

  if (getCmdOption(input_str, input_str + in_num, "-tensor")) {
    tensor = getCmdOption(input_str, input_str + in_num, "-tensor");
  } else {
    tensor = "p";
  }
  if (getCmdOption(input_str, input_str + in_num, "-method")) {
    method = atoi(getCmdOption(input_str, input_str + in_num, "-method"));
  }
  if (getCmdOption(input_str, input_str + in_num, "-ppmethod")) {
    ppmethod = atoi(getCmdOption(input_str, input_str + in_num, "-ppmethod"));
  }
  if (getCmdOption(input_str, input_str + in_num, "-seed")) {
    seed = atoi(getCmdOption(input_str, input_str + in_num, "-seed"));
  }
  if (getCmdOption(input_str, input_str + in_num, "-msdt")) {
    int msdt = atoi(getCmdOption(input_str, input_str + in_num, "-msdt"));
    if (msdt > 0)
      use_msdt = true;
  } else {
    use_msdt = false;
  }
  if (getCmdOption(input_str, input_str + in_num, "-ppoperator")) {
    int ppoperator =
        atoi(getCmdOption(input_str, input_str + in_num, "-ppoperator"));
    if (ppoperator > 0)
      renew_ppoperator = true;
  }
  if (getCmdOption(input_str, input_str + in_num, "-update_percentage_pp")) {
    update_percentage_pp = atof(
        getCmdOption(input_str, input_str + in_num, "-update_percentage_pp"));
    if (update_percentage_pp < 0 || update_percentage_pp > 1)
      update_percentage_pp = 1.0;
  } else {
    update_percentage_pp = 1.0;
  }
  if (getCmdOption(input_str, input_str + in_num, "-dim")) {
    dim = atoi(getCmdOption(input_str, input_str + in_num, "-dim"));
    if (dim < 0)
      dim = 3;
  } else {
    dim = 3;
  }
  if (getCmdOption(input_str, input_str + in_num, "-maxsweep")) {
    maxsweep = atoi(getCmdOption(input_str, input_str + in_num, "-maxsweep"));
    if (maxsweep < 0)
      maxsweep = 5e3;
  } else {
    maxsweep = 5e3;
  }
  if (getCmdOption(input_str, input_str + in_num, "-timelimit")) {
    timelimit = atof(getCmdOption(input_str, input_str + in_num, "-timelimit"));
    if (timelimit < 0)
      timelimit = 5e7;
  } else {
    timelimit = 5e7;
  }
  if (getCmdOption(input_str, input_str + in_num, "-sizes")) {
    sizes = getVectorCmdOption(input_str, input_str + in_num, "-sizes");
  }
  if (getCmdOption(input_str, input_str + in_num, "-mesh")) {
    processor_mesh = getVectorCmdOption(input_str, input_str + in_num, "-mesh");
  }
  if (getCmdOption(input_str, input_str + in_num, "-rank")) {
    R = atoi(getCmdOption(input_str, input_str + in_num, "-rank"));
  }
  if (getCmdOption(input_str, input_str + in_num, "-issparse")) {
    issparse = atoi(getCmdOption(input_str, input_str + in_num, "-issparse"));
    if (issparse < 0 || issparse > 1)
      issparse = 0;
  } else {
    issparse = 0;
  }
  if (getCmdOption(input_str, input_str + in_num, "-resprint")) {
    resprint = atoi(getCmdOption(input_str, input_str + in_num, "-resprint"));
    if (resprint < 0)
      resprint = 10;
  } else {
    resprint = 10;
  }
  if (getCmdOption(input_str, input_str + in_num, "-tol")) {
    tol = atof(getCmdOption(input_str, input_str + in_num, "-tol"));
    if (tol < 0 || tol > 1)
      tol = 1e-10;
  } else {
    tol = 1e-10;
  }
  if (getCmdOption(input_str, input_str + in_num, "-pp_res_tol")) {
    pp_res_tol =
        atof(getCmdOption(input_str, input_str + in_num, "-pp_res_tol"));
    if (pp_res_tol < 0 || pp_res_tol > 1)
      pp_res_tol = 1e-2;
  } else {
    pp_res_tol = 1e-2;
  }
  if (getCmdOption(input_str, input_str + in_num, "-lambda")) {
    lambda_ = atof(getCmdOption(input_str, input_str + in_num, "-lambda"));
    if (lambda_ < 0)
      lambda_ = 0.;
  } else {
    lambda_ = 0.;
  }
  if (getCmdOption(input_str, input_str + in_num, "-magni")) {
    magni = atof(getCmdOption(input_str, input_str + in_num, "-magni"));
    if (magni < 0)
      magni = 1.;
  } else {
    magni = 1.;
  }
  if (getCmdOption(input_str, input_str + in_num, "-filename")) {
    filename = getCmdOption(input_str, input_str + in_num, "-filename");
  } else {
    filename = "out.csv";
  }
  if (getCmdOption(input_str, input_str + in_num, "-tensorfile")) {
    tensorfile = getCmdOption(input_str, input_str + in_num, "-tensorfile");
  } else {
    tensorfile = "test";
  }
  if (getCmdOption(input_str, input_str + in_num, "-colmin")) {
    col_min = atof(getCmdOption(input_str, input_str + in_num, "-colmin"));
  } else {
    col_min = 0.5;
  }
  if (getCmdOption(input_str, input_str + in_num, "-colmax")) {
    col_max = atof(getCmdOption(input_str, input_str + in_num, "-colmax"));
  } else {
    col_max = 0.9;
  }
  if (getCmdOption(input_str, input_str + in_num, "-rationoise")) {
    ratio_noise =
        atof(getCmdOption(input_str, input_str + in_num, "-rationoise"));
    if (ratio_noise < 0)
      ratio_noise = 0.01;
  } else {
    ratio_noise = 0.01;
  }

  {
    double start_time = MPI_Wtime();
    World dw(argc, argv);
    World sworld(MPI_COMM_SELF);
    srand48(dw.rank * 1);
    IASSERT(sizes.size() == dim);

    if (dw.rank == 0) {
      cout << "  tensor=  " << tensor << "  method=  " << method << "  ppmethod=  " << ppmethod << "  seed=  " << seed << endl;
      cout << "  dim=  " << dim << "  rank=  " << R
           << "  use_msdt=  " << use_msdt
           << "  renew_ppoperator=  " << renew_ppoperator << endl;
      cout << "  issparse=  " << issparse << "  tolerance=  " << tol
           << "  restarttol=  " << pp_res_tol << endl;
      cout << "  lambda=  " << lambda_ << "  magnitude=  " << magni
           << "  filename=  " << filename << endl;
      cout << "  col_min=  " << col_min << "  col_max=  " << col_max
           << "  rationoise  " << ratio_noise << endl;
      cout << "  timelimit=  " << timelimit << "  maxsweep=  " << maxsweep
           << "  resprint=  " << resprint << endl;
      cout << "  tensorfile=  " << tensorfile
           << "  update_percentage_pp=  " << update_percentage_pp << endl;
      cout << "  sizes=  ";
      for (int i = 0; i < dim; i++) {
        cout << sizes[i] << "  ";
      }
      cout << endl;
      cout << "  processor_mesh=  ";
      for (int i = 0; i < processor_mesh.size(); i++) {
        cout << processor_mesh[i] << "  ";
      }
      cout << endl;
    }

    // initialization of tensor
    Tensor<> V;

    if (tensor[0] == 'c') {
      // c : designed tensor with constrained collinearity
      int lens[dim];
      for (int i = 0; i < dim; i++)
        lens[i] = sizes[i];
      gen_collinearity(V, lens, dim, R, col_min, col_max, seed, dw, processor_mesh);
    } else if (tensor[0] == 'r') {
      if (strlen(tensor) > 1 && tensor[1] == '2') {
        // r2 : random tensor
        int lens[dim];
        for (int i = 0; i < dim; i++)
          lens[i] = sizes[i];
        // create subworld tensor
        Tensor<> *V_subworld = NULL;
        if (dw.rank == 0) {
          V_subworld = new Tensor<>(dim, issparse, lens, sworld);
          V_subworld->fill_random(0., 1.);
        }
        V = Tensor<>(dim, issparse, lens, dw);
        V.add_from_subworld(V_subworld);
        delete V_subworld;
      } else {
        // r : tensor made by random matrices
        int lens[dim];
        for (int i = 0; i < dim; i++)
          lens[i] = sizes[i];
        Matrix<> **W = (Matrix<> **)malloc(
            dim * sizeof(Matrix<> *)); // N matrices V will be decomposed into
        for (int i = 0; i < dim; i++) {
          // use subworld matrix to make the matrix deterministic across various
          // processes
          Matrix<> *W_subworld = NULL;
          if (dw.rank == 0) {
            W_subworld = new Matrix<>(sizes[i], R, sworld);
            W_subworld->fill_random(0., 1.);
          }
          W[i] = new Matrix<>(sizes[i], R, dw);
          W[i]->add_from_subworld(W_subworld);
          delete W_subworld;
        }
        build_V(V, W, dim, dw, processor_mesh);
        delete[] W;
      }
    } else if (tensor[0] == 'o') {
      // o1 : coil-100 dataset Rank=20 suggested
      if (strlen(tensor) > 1 && tensor[1] == '1') {
        tensorfile = "coil-100.bin";
        MPI_File_open(MPI_COMM_WORLD, tensorfile,
                      MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
        int lens[dim];
        lens[0] = 3;
        lens[1] = 128;
        lens[2] = 128;
        lens[3] = 7200;
        // for (int i=0; i<dim; i++) lens[i]=s;
        V = Tensor<>(dim, issparse, lens, dw);
        if (dw.rank == 0)
          cout << "Read the tensor from file coil-100 ...... " << endl;
        V.read_dense_from_file(fh);
        if (dw.rank == 0)
          cout << "Read coil-100 dataset finished " << endl;
        // V.print();
      }
      // o2 : time-lapse dataset Rank=32 suggested
      else if (strlen(tensor) > 1 && tensor[1] == '2') {
        tensorfile = "time-lapse.bin";
        MPI_File_open(MPI_COMM_WORLD, tensorfile,
                      MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
        int lens[dim];
        lens[0] = 33;
        lens[1] = 1344;
        lens[2] = 1024;
        lens[3] = 9;
        // for (int i=0; i<dim; i++) lens[i]=s;
        V = Tensor<>(dim, issparse, lens, dw);
        if (dw.rank == 0)
          cout << "Read the tensor from file time-lapse ...... " << endl;
        V.read_dense_from_file(fh);
        if (dw.rank == 0)
          cout << "Read time-lapse dataset finished " << endl;
        // V.print();
      }
    }

    double Vnorm = V.norm2();
    if (dw.rank == 0) {
      cout << "Vnorm= " << Vnorm << endl;
    }
    ofstream Plot_File(filename);

    Matrix<> **W = (Matrix<> **)malloc(
        V.order * sizeof(Matrix<> *)); // N matrices V will be decomposed into
    Matrix<> *grad_W = new Matrix<>[V.order]; // gradients in N dimensions
    for (int i = 0; i < V.order; i++) {
      Matrix<> *W_subworld = NULL;
      if (dw.rank == 0) {
        W_subworld = new Matrix<>(V.lens[i], R, sworld);
        W_subworld->fill_random(0., 1.);
      }
      W[i] = new Matrix<>(V.lens[i], R, dw);
      grad_W[i] = Matrix<>(V.lens[i], R, dw);
      W[i]->add_from_subworld(W_subworld);
      delete W_subworld;
    }

    // V.write_dense_to_file (fh);
    int lens[dim];
    for (int i = 0; i < dim; i++)
      lens[i] = V.lens[i];

    if (method == 0) {
      if (dw.rank == 0) {
        cout << "============CPSimpleOptimizer=============" << endl;
      }
      CPD<double, CPSimpleOptimizer<double>> decom(dim, lens, R, dw);
      decom.Init(&V, W);
      decom.als(tol, Vnorm, timelimit, maxsweep, resprint, Plot_File);
    } else if (method == 1) {
      if (dw.rank == 0) {
        cout << "============CPLocalOptimizer=============" << endl;
      }
      CPD<double, CPLocalOptimizer<double>> decom(dim, lens, R, dw);
      decom.Init(&V, W);
      decom.als(tol, Vnorm, timelimit, maxsweep, resprint, Plot_File);
    } else if (method == 2) {
      if (dw.rank == 0) {
        cout << "============CPDTOptimizer=============" << endl;
      }
      CPD<double, CPDTOptimizer<double>> decom(dim, lens, R, dw, use_msdt);
      decom.Init(&V, W);
      decom.als(tol, Vnorm, timelimit, maxsweep, resprint, Plot_File);
    } else if (method == 3) {
      if (dw.rank == 0) {
        cout << "============CPDTLocalOptimizer=============" << endl;
      }
      CPD<double, CPDTLocalOptimizer<double>> decom(dim, lens, R, dw, use_msdt);
      decom.Init(&V, W);
      decom.als(tol, Vnorm, timelimit, maxsweep, resprint, Plot_File);
    } else if (method == 4) {
      if (dw.rank == 0) {
        cout << "============CPPPOptimizer=============" << endl;
      }
      CPD<double, CPPPOptimizer<double>> decom(dim, lens, R, dw, pp_res_tol,
                                               use_msdt, renew_ppoperator, ppmethod);
      decom.Init(&V, W);
      decom.als(tol, Vnorm, timelimit, maxsweep, resprint, Plot_File);
    } else if (method == 5) {
      if (dw.rank == 0) {
        cout << "============CPPPLocalOptimizer=============" << endl;
      }
      CPD<double, CPPPLocalOptimizer<double>> decom(
          dim, lens, R, dw, pp_res_tol, use_msdt, renew_ppoperator, ppmethod);
      decom.Init(&V, W);
      decom.als(tol, Vnorm, timelimit, maxsweep, resprint, Plot_File);
    }

    if (dw.rank == 0) {
      printf("experiment took %lf seconds\n", MPI_Wtime() - start_time);
    }

    if (tensor[0] == 'o') {
      MPI_File_close(&fh);
    }
  }

  MPI_Finalize();
  return 0;
}

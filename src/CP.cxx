
#include "utils/common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype, class Optimizer>
CPD<dtype, Optimizer>::CPD(int order, int size_, int r, World &dw)
    : Decomposition<dtype>(order, size_, r, dw) {
  optimizer = new Optimizer(order, r, dw);

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }
}

template <typename dtype, class Optimizer>
CPD<dtype, Optimizer>::CPD(int order, int *size_, int r, World &dw)
    : Decomposition<dtype>(order, size_, r, dw) {

  for (int i = 1; i < order; i++) {
    assert(this->size[i] == size_[i]);
    assert(this->rank[i] == r);
  }

  optimizer = new Optimizer(order, r, dw);

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }
}

template <typename dtype, class Optimizer>
CPD<dtype, Optimizer>::CPD(int order, int *size_, int r, World &dw,
                           double tol_restart_dt, bool use_msdt,
                           bool renew_ppoperator, int ppmethod)
    : Decomposition<dtype>(order, size_, r, dw) {

  for (int i = 1; i < order; i++) {
    assert(this->size[i] == size_[i]);
    assert(this->rank[i] == r);
  }

  optimizer =
      new Optimizer(order, r, dw, tol_restart_dt, use_msdt, renew_ppoperator, ppmethod);

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }
}

template <typename dtype, class Optimizer>
CPD<dtype, Optimizer>::CPD(int order, int *size_, int r, World &dw,
                           bool use_msdt)
    : Decomposition<dtype>(order, size_, r, dw) {

  for (int i = 1; i < order; i++) {
    assert(this->size[i] == size_[i]);
    assert(this->rank[i] == r);
  }

  optimizer = new Optimizer(order, r, dw, use_msdt);

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }
}

template <typename dtype, class Optimizer>
void CPD<dtype, Optimizer>::Init(Tensor<dtype> *input, Matrix<dtype> **mat,
                                 double lambda) {

  Decomposition<dtype>::Init(input, mat);
  World *dw = this->world;
  // initialize grad_W
  if (grad_W != NULL) {
    delete[] grad_W;
  }
  grad_W = new Matrix<>[this->order];
  for (int i = 0; i < this->order; i++) {
    grad_W[i] = Matrix<dtype>(this->size[i], this->rank[i], *dw);
    grad_W[i].fill_random(0, 1);
  }
  // configure the optimizer
  this->optimizer->configure(input, mat, grad_W, lambda);
}

template <typename dtype, class Optimizer>
void CPD<dtype, Optimizer>::print_grad(int i) const {
  assert(grad_W != NULL);
  grad_W[i].print();
}

template <typename dtype, class Optimizer> CPD<dtype, Optimizer>::~CPD() {
  if (grad_W != NULL) {
    delete[] grad_W;
  }
  if (optimizer != NULL) {
    delete optimizer;
  }
}

template <typename dtype, class Optimizer>
void CPD<dtype, Optimizer>::update_gradnorm() {
  gradnorm = 0;
  for (int i = 0; i < this->order; i++) {
    gradnorm += this->grad_W[i].norm2();
  }
}

template <typename dtype, class Optimizer>
bool CPD<dtype, Optimizer>::als(double tol, double Vnorm, double timelimit,
                                int maxsweep, int resprint, ofstream &Plot_File,
                                bool bench) {

  Timer_epoch tALS("ALS");
  tALS.begin();

  cout.precision(13);

  World *dw = this->world;
  double st_time = MPI_Wtime();
  int iters = 0;
  double sweeps = 0;
  double diffnorm_V = 1000.;
  double fitness = 0.;

  if (bench == false) {
    if (dw->rank == 0)
      Plot_File << "[dim],[iter],[gradnorm],[tol],[pp_update],[fitness],[dtime]"
                << "\n"; // Headings for file
  }

  while (fabs(sweeps - maxsweep) > 1e-5 && sweeps < maxsweep) {
    // print the gradient norm
    if (sweeps - int(sweeps) == 0 &&
        (int(sweeps) % resprint == 0 || sweeps == 0)) {
      double st_time1 = MPI_Wtime();
      update_gradnorm();
      // residual
      if (sweeps == 0) {
        fitness = 0.;
      } else {
        this->optimizer->update_S_residual_calc();
        double Wnorms = sqrt(this->optimizer->S.reduce(CTF::OP_SUM));
        Matrix<> temp =
            Matrix<>(this->size[this->order - 1], this->rank[this->order - 1]);
        temp["ij"] = this->optimizer->M[this->order - 1]->operator[]("ij") *
                     this->optimizer->W[this->order - 1]->operator[]("ij");
        double T_W_inner = temp.reduce(CTF::OP_SUM);
        diffnorm_V = sqrt(Vnorm * Vnorm + Wnorms * Wnorms - 2. * T_W_inner);
        fitness = 1. - diffnorm_V / Vnorm;
      }
      // record time
      st_time += MPI_Wtime() - st_time1;
      double dtime = MPI_Wtime() - st_time;
      if (bench == false) {
        if (dw->rank == 0) {
          cout << "  [dim]=  " << (this->V)->lens[0]
               << "  [sweeps]=  " << sweeps << "  [gradnorm]  " << gradnorm
               << "  [tol]  " << tol << "  [pp_update]  " << 0
               << "  [fitness]  " << fitness << "  [dtime]  " << dtime << "\n";
          Plot_File << (this->V)->lens[0] << "," << sweeps << "," << gradnorm
                    << "," << tol << "," << 0 << "," << fitness << "," << dtime
                    << "\n";
          // flush the contents to csv
          if (iters % 100 == 0 && iters != 0) {
            Plot_File << endl;
          }
        }
      } else {
        if (dw->rank == 0 && int(iters) != 0) {
          cout << "  [dimension tree step time]  " << dtime << "\n";
          Plot_File << "[DTtime]"
                    << "," << dtime << "\n";
        }
      }
      if ((gradnorm < tol) || MPI_Wtime() - st_time > timelimit)
        break;
    }

    Timer tals_step("als-step");
    tals_step.start();
    sweeps += this->optimizer->step();
    iters += 1;
    tals_step.stop();

    // Normalize(this->W, this->order, *dw);
    // print .
    if (iters % 10 == 0 && dw->rank == 0)
      printf(".");
  }
  if (dw->rank == 0) {
    printf("\nIters = %d Final proj-grad norm %E \n", iters, gradnorm);
    printf("tf took %lf seconds\n", MPI_Wtime() - st_time);
  }
  if (bench == false) {
    Plot_File.close();
  }

  tALS.end();

  if (sweeps == maxsweep + 1)
    return false;
  else
    return true;
}

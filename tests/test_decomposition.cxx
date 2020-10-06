
#include "../include/cpd.hpp"
#include "test_local_mttkrp.cxx"
#include <ctf.hpp>

using namespace CTF;

void TEST_CPD(World &dw) {
  if (dw.rank == 0) {
    cout << "Test CPD" << endl;
  }

  // test init
  int lens[3];
  int size = 8;
  for (int i = 0; i < 3; i++)
    lens[i] = size + 3 * i;

  // test dimension
  bool renew_ppoperator = false;
  int ppmethod = 0;
  CPD<double, CPSimpleOptimizer<double>> decom(3, lens, 5, dw);
  CPD<double, CPLocalOptimizer<double>> decom_local(3, lens, 5, dw);
  CPD<double, CPDTOptimizer<double>> decom_dt(3, lens, 5, dw, false);
  CPD<double, CPDTOptimizer<double>> decom_msdt(3, lens, 5, dw, true);
  CPD<double, CPDTLocalOptimizer<double>> decom_dt_local(3, lens, 5, dw, false);
  CPD<double, CPDTLocalOptimizer<double>> decom_msdt_local(3, lens, 5, dw,
                                                           true);
  CPD<double, CPPPOptimizer<double>> decom_pp(3, lens, 5, dw, 1e-5, true,
                                              renew_ppoperator, ppmethod);
  CPD<double, CPPPLocalOptimizer<double>> decom_pp_local(
      3, lens, 5, dw, 1e-5, true, renew_ppoperator, ppmethod);

  assert(decom.order == 3);
  assert(decom.rank[0] == 5);

  assert(decom_dt.order == 3);
  assert(decom_dt.rank[0] == 5);

  assert(decom_msdt.order == 3);
  assert(decom_msdt.rank[0] == 5);

  assert(decom_local.order == 3);
  assert(decom_local.rank[0] == 5);

  assert(decom_dt_local.order == 3);
  assert(decom_dt_local.rank[0] == 5);

  assert(decom_msdt_local.order == 3);
  assert(decom_msdt_local.rank[0] == 5);

  assert(decom_pp.order == 3);
  assert(decom_pp.rank[0] == 5);

  assert(decom_pp_local.order == 3);
  assert(decom_pp_local.rank[0] == 5);

  Tensor<> *V = new Tensor<>(3, lens, dw);
  V->fill_random(0, 1);
  double Vnorm = V->norm2();

  Matrix<> **W = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_dt = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_msdt = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_dt_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_msdt_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_pp = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_pp_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));

  for (int i = 0; i < 3; i++) {
    W[i] = new Matrix<>(lens[i], 5, dw);
    W_local[i] = new Matrix<>(lens[i], 5, dw);
    W_dt[i] = new Matrix<>(lens[i], 5, dw);
    W_msdt[i] = new Matrix<>(lens[i], 5, dw);
    W_dt_local[i] = new Matrix<>(lens[i], 5, dw);
    W_msdt_local[i] = new Matrix<>(lens[i], 5, dw);
    W_pp[i] = new Matrix<>(lens[i], 5, dw);
    W_pp_local[i] = new Matrix<>(lens[i], 5, dw);

    W[i]->fill_random(0, 1);
    W_local[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_dt[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_msdt[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_dt_local[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_msdt_local[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_pp[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_pp_local[i]->operator[]("ij") = W[i]->operator[]("ij");
  }

  ofstream Plot_File("results/test.csv");

  decom.Init(V, W);
  decom.als(1e-5, Vnorm, 1000, 4, 100, Plot_File);

  decom_dt.Init(V, W_dt);
  decom_dt.als(1e-5, Vnorm, 1000, 4, 100, Plot_File);

  decom_msdt.Init(V, W_msdt);
  decom_msdt.als(1e-5, Vnorm, 1000, 4, 100, Plot_File);

  decom_local.Init(V, W_local);
  decom_local.als(1e-5, Vnorm, 1000, 4, 100, Plot_File);

  decom_dt_local.Init(V, W_dt_local);
  decom_dt_local.als(1e-5, Vnorm, 1000, 4, 100, Plot_File);

  decom_msdt_local.Init(V, W_msdt_local);
  decom_msdt_local.als(1e-5, Vnorm, 1000, 4, 100, Plot_File);

  decom_pp.Init(V, W_pp);
  decom_pp.als(1e-5, Vnorm, 1000, 4, 100, Plot_File);

  decom_pp_local.Init(V, W_pp_local);
  decom_pp_local.als(1e-5, Vnorm, 1000, 4, 100, Plot_File);

  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(lens[i], 5, dw);
    diff["ij"] = W[i]->operator[]("ij") - W_dt[i]->operator[]("ij");
    double diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_msdt[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_local[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_dt_local[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_msdt_local[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_pp[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_pp_local[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);
  }
}

void TEST_PP(World &dw) {
  if (dw.rank == 0) {
    cout << "Test PP" << endl;
  }

  // test init
  int lens[3];
  int size = 8;
  for (int i = 0; i < 3; i++)
    lens[i] = size + 3 * i;

  bool renew_ppoperator = false;
  int ppmethod = 0;
  // test dimension
  CPD<double, CPSimpleOptimizer<double>> decom(3, lens, 5, dw);
  CPD<double, CPPPOptimizer<double>> decom_pp(3, lens, 5, dw, 10.0, false,
                                              renew_ppoperator, ppmethod);
  CPD<double, CPPPLocalOptimizer<double>> decom_pp_local(
      3, lens, 5, dw, 10.0, false, renew_ppoperator, ppmethod);

  assert(decom.order == 3);
  assert(decom.rank[0] == 5);

  assert(decom_pp.order == 3);
  assert(decom_pp.rank[0] == 5);

  assert(decom_pp_local.order == 3);
  assert(decom_pp_local.rank[0] == 5);

  Tensor<> *V = new Tensor<>(3, lens, dw);
  V->fill_random(0, 1);
  double Vnorm = V->norm2();

  Matrix<> **W = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_pp = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_pp_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));

  for (int i = 0; i < 3; i++) {
    W[i] = new Matrix<>(lens[i], 5, dw);
    W_pp[i] = new Matrix<>(lens[i], 5, dw);
    W_pp_local[i] = new Matrix<>(lens[i], 5, dw);

    W[i]->fill_random(0, 1);
    W_pp[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_pp_local[i]->operator[]("ij") = W[i]->operator[]("ij");
  }

  ofstream Plot_File("results/test.csv");

  decom.Init(V, W);
  decom.als(1e-5, Vnorm, 1000, 2, 100, Plot_File);

  decom_pp.Init(V, W_pp);
  decom_pp.als(1e-5, Vnorm, 1000, 2, 100, Plot_File);

  decom_pp_local.Init(V, W_pp_local);
  decom_pp_local.als(1e-5, Vnorm, 1000, 2, 100, Plot_File);

  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(lens[i], 5, dw);
    diff["ij"] = W[i]->operator[]("ij") - W_pp[i]->operator[]("ij");
    double diff_norm = diff.norm2();
    if (i == 0 || i == 1) {
      assert(diff_norm < 1e-8);
    }

    diff["ij"] = W[i]->operator[]("ij") - W_pp_local[i]->operator[]("ij");
    diff_norm = diff.norm2();
    if (i == 0 || i == 1) {
      assert(diff_norm < 1e-8);
    }
  }

  decom_pp.als(1e-5, Vnorm, 1000, 2, 100, Plot_File);
  decom_pp_local.als(1e-5, Vnorm, 1000, 2, 100, Plot_File);
  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(lens[i], 5, dw);
    diff["ij"] = W_pp_local[i]->operator[]("ij") - W_pp[i]->operator[]("ij");
    double diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);
  }
}

void TEST_PP_local(World &dw) {
  if (dw.rank == 0) {
    cout << "Test PP_local" << endl;
  }

  // test init
  int lens[3];
  int size = 8;
  for (int i = 0; i < 3; i++)
    lens[i] = size + 3 * i;

  bool renew_ppoperator = false;
  int ppmethod = 0;
  // test dimension
  CPD<double, CPPPOptimizer<double>> decom_pp(3, lens, 5, dw, 1., false,
                                              renew_ppoperator, ppmethod);
  CPD<double, CPPPLocalOptimizer<double>> decom_pp_local(
      3, lens, 5, dw, 1., false, renew_ppoperator, ppmethod);

  Tensor<> *V = new Tensor<>(3, lens, dw);
  V->fill_random(0, 1);
  double Vnorm = V->norm2();

  Matrix<> **W_pp = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_pp_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));

  for (int i = 0; i < 3; i++) {
    W_pp[i] = new Matrix<>(lens[i], 5, dw);
    W_pp_local[i] = new Matrix<>(lens[i], 5, dw);

    W_pp[i]->fill_random(0, 1);
    W_pp_local[i]->operator[]("ij") = W_pp[i]->operator[]("ij");
  }

  ofstream Plot_File("results/test.csv");

  decom_pp.Init(V, W_pp);
  decom_pp.als(1e-5, Vnorm, 1000, 10, 100, Plot_File);

  decom_pp_local.Init(V, W_pp_local);
  decom_pp_local.als(1e-5, Vnorm, 1000, 10, 100, Plot_File);

  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(lens[i], 5, dw);
    diff["ij"] = W_pp_local[i]->operator[]("ij") - W_pp[i]->operator[]("ij");
    double diff_norm = diff.norm2();
    assert(diff_norm < 1e-4);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  World dw(argc, argv);

  TEST_local_mttkrp(dw);
  TEST_CPD(dw);
  TEST_PP(dw);
  TEST_PP_local(dw);

  if (dw.rank == 0) {
    cout << "All tests passed" << endl;
  }
  MPI_Finalize();
}

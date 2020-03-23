
#include "../include/cpd.hpp"
#include "test_local_mttkrp.cxx"
#include <ctf.hpp>

using namespace CTF;

void TEST_decomposition(World &dw) {

  // test dimension
  Decomposition<double> decom(3, 5, 2, dw);
  cout << decom.order << endl;
  assert(decom.order == 3);
  assert(decom.rank[0] == 2);

  // test init
  int lens[3];
  for (int i = 0; i < 3; i++)
    lens[i] = 5;
  Tensor<> *V = new Tensor<>(3, lens, dw);
  V->fill_random(0, 1);

  Matrix<> **W = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  for (int i = 0; i < 3; i++) {
    W[i] = new Matrix<>(5, 2, dw);
    W[i]->fill_random(0, 1);
  }
  decom.Init(V, W);
  decom.print_W(0);
  decom.print_W(1);
}

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
  CPD<double, CPSimpleOptimizer<double>> decom(3, lens, 5, dw);
  CPD<double, CPDTOptimizer<double>> decom_dt(3, lens, 5, dw);
  CPD<double, CPDTLocalOptimizer<double>> decom_dt_local(3, lens, 5, dw);
  CPD<double, CPLocalOptimizer<double>> decom_local(3, lens, 5, dw);
  CPD<double, CPPPOptimizer<double>> decom_pp(3, lens, 5, dw, 1e-5);

  assert(decom.order == 3);
  assert(decom.rank[0] == 5);

  assert(decom_dt.order == 3);
  assert(decom_dt.rank[0] == 5);

  assert(decom_local.order == 3);
  assert(decom_local.rank[0] == 5);

  assert(decom_dt_local.order == 3);
  assert(decom_dt_local.rank[0] == 5);

  assert(decom_pp.order == 3);
  assert(decom_pp.rank[0] == 5);

  Tensor<> *V = new Tensor<>(3, lens, dw);
  V->fill_random(0, 1);

  Matrix<> **W = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_dt = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_dt_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_pp = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));

  for (int i = 0; i < 3; i++) {
    W[i] = new Matrix<>(lens[i], 5, dw);
    W_dt[i] = new Matrix<>(lens[i], 5, dw);
    W_local[i] = new Matrix<>(lens[i], 5, dw);
    W_dt_local[i] = new Matrix<>(lens[i], 5, dw);
    W_pp[i] = new Matrix<>(lens[i], 5, dw);

    W[i]->fill_random(0, 1);
    W_dt[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_local[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_dt_local[i]->operator[]("ij") = W[i]->operator[]("ij");
    W_pp[i]->operator[]("ij") = W[i]->operator[]("ij");
  }

  ofstream Plot_File("results/test.csv");

  decom.Init(V, W);
  decom.als(1e-5, 1000, 3, 100, Plot_File);

  decom_dt.Init(V, W_dt);
  decom_dt.als(1e-5, 1000, 3, 100, Plot_File);

  decom_local.Init(V, W_local);
  decom_local.als(1e-5, 1000, 3, 100, Plot_File);

  decom_dt_local.Init(V, W_dt_local);
  decom_dt_local.als(1e-5, 1000, 3, 100, Plot_File);

  decom_pp.Init(V, W_pp);
  decom_pp.als(1e-5, 1000, 3, 100, Plot_File);

  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(lens[i], 5, dw);
    diff["ij"] = W[i]->operator[]("ij") - W_dt[i]->operator[]("ij");
    double diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_local[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_dt_local[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);

    diff["ij"] = W[i]->operator[]("ij") - W_pp[i]->operator[]("ij");
    diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);
  }
}

int main(int argc, char **argv) {
  int logn;
  int64_t n;

  MPI_Init(&argc, &argv);

  World dw(argc, argv);

  TEST_local_mttkrp(dw);
  TEST_CPD(dw);

  if (dw.rank == 0) {
    cout << "All tests passed" << endl;
  }
  MPI_Finalize();
}

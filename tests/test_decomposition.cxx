
#include "../include/cpd.hpp"
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
  Matrix<> *W = new Matrix<>[3];
  for (int i = 0; i < 3; i++) {
    W[i] = Matrix<>(5, 2, dw);
    W[i].fill_random(0, 1);
  }
  decom.Init(V, W);
  decom.print_W(0);
  decom.print_W(1);
}

void TEST_CPD(World &dw) {
  cout << "Test CPD" << endl;

  // test dimension
  CPD<double, CPSimpleOptimizer<double>> decom(3, 13, 5, dw);
  CPD<double, CPDTOptimizer<double>> decom_dt(3, 13, 5, dw);

  assert(decom.order == 3);
  assert(decom.rank[0] == 5);

  assert(decom_dt.order == 3);
  assert(decom_dt.rank[0] == 5);

  // test init
  int lens[3];
  for (int i = 0; i < 3; i++)
    lens[i] = 13;

  Tensor<> *V = new Tensor<>(3, lens, dw);
  V->fill_random(0, 1);

  Matrix<> *W = new Matrix<>[3];
  Matrix<> *W_dt = new Matrix<>[3];

  for (int i = 0; i < 3; i++) {
    W[i] = Matrix<>(13, 5, dw);
    W_dt[i] = Matrix<>(13, 5, dw);

    W[i].fill_random(0, 1);
    W_dt[i]["ij"] = W[i]["ij"];
  }

  ofstream Plot_File("results/test.csv");

  decom.Init(V, W);
  decom.als(1e-5, 1000, 10, 100, Plot_File);

  decom_dt.Init(V, W_dt);
  decom_dt.als(1e-5, 1000, 10, 100, Plot_File);

  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(13, 5, dw);
    diff["ij"] = W[i]["ij"] - W_dt[i]["ij"];
    double diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);
  }
}

int main(int argc, char **argv) {
  int logn;
  int64_t n;

  MPI_Init(&argc, &argv);

  World dw(argc, argv);

  // TEST_decomposition(dw);
  TEST_CPD(dw);

  cout << "All tests passed" << endl;

  MPI_Finalize();
}

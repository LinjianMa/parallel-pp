
#include "../include/cpd.hpp"
#include <ctf.hpp>

using namespace CTF;

void TEST_local_mttkrp(World &dw) {
  cout << "Test local mttkrp" << endl;

  // test init
  int lens[3];
  for (int i = 0; i < 3; i++)
    lens[i] = 8;

  Tensor<> *V = new Tensor<>(3, lens, dw);
  V->fill_random(0, 1);

  Matrix<> *W = new Matrix<>[3];
  Matrix<> *W_local = new Matrix<>[3];

  for (int i = 0; i < 3; i++) {
    W[i] = Matrix<>(8, 5, dw);
    W_local[i] = Matrix<>(8, 5, dw);

    W[i].fill_random(0, 1);
    W_local[i]["ij"] = W[i]["ij"];
  }

  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(8, 5, dw);

    MTTKRP(V, W_local, i);

    W[i].print();
    W_local[i].print();

    diff["ij"] = W[i]["ij"] - W_local[i]["ij"];
    double diff_norm = diff.norm2();
    if (dw.rank == 0) {
      cout << "diff norm is: " << diff_norm << endl;
    }
    assert(diff_norm < 1e-8);
  }
}

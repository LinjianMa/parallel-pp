
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

  Matrix<> **W = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));

  for (int i = 0; i < 3; i++) {
    W[i] = new Matrix<>(8, 5, dw);
    W_local[i] = new Matrix<>(8, 5, dw);

    W[i]->fill_random(0, 1);
    W_local[i]->operator[]("ij") = W[i]->operator[]("ij");
  }

  LocalMTTKRP<double> *local_mttkrp = new LocalMTTKRP<double>(3, 5, dw);
  local_mttkrp->setup(V, W_local);
  local_mttkrp->distribute_W();

  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(8, 5, dw);

    local_mttkrp->post_mttkrp_reduce(i);

    diff["ij"] = W[i]->operator[]("ij") - W_local[i]->operator[]("ij");
    double diff_norm = diff.norm2();
    if (dw.rank == 0) {
      cout << "diff norm is: " << diff_norm << endl;
    }
    assert(diff_norm < 1e-8);
  }
}

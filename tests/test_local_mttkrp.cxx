
#include "../include/cpd.hpp"
#include <ctf.hpp>

using namespace CTF;

void TEST_local_mttkrp(World &dw) {
  if (dw.rank == 0) {
    cout << "Test local mttkrp" << endl;
  }

  // test init
  int size = 8;
  int rank = 5;
  int lens[3];
  for (int i = 0; i < 3; i++)
    lens[i] = size;

  Tensor<> *V = new Tensor<>(3, lens, dw);
  V->fill_random(0, 1);

  Matrix<> **W = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));
  Matrix<> **W_local = (Matrix<> **)malloc(3 * sizeof(Matrix<> *));

  for (int i = 0; i < 3; i++) {
    W[i] = new Matrix<>(size, rank, dw);
    W_local[i] = new Matrix<>(size, rank, dw);

    W[i]->fill_random(0, 1);
    W_local[i]->operator[]("ij") = W[i]->operator[]("ij");
  }

  LocalMTTKRP<double> *local_mttkrp = new LocalMTTKRP<double>(3, rank, dw);
  local_mttkrp->setup(V, W_local);
  for (int i = 0; i < 3; i++) {
    local_mttkrp->distribute_W(i, local_mttkrp->W, local_mttkrp->W_local);
  }
  local_mttkrp->setup_V_local_data();
  local_mttkrp->construct_mttkrp_locals();

  for (int i = 0; i < V->order; i++) {
    Matrix<> diff = Matrix<>(size, rank, dw);

    local_mttkrp->post_mttkrp_reduce(i);

    diff["ij"] = W[i]->operator[]("ij") - W_local[i]->operator[]("ij");
    double diff_norm = diff.norm2();
    assert(diff_norm < 1e-8);
  }
  if (dw.rank == 0) {
    cout << "Local mttkrp test passed" << endl;
  }
}

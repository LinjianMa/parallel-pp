
// #include "decomposition.h"
#include "../utils/common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPLocalOptimizer<dtype>::CPLocalOptimizer(int order, int r, World &dw)
    : CPOptimizer<dtype>(order, r, dw) {

  // make the char seq_V
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }

  local_mttkrp = new LocalMTTKRP<dtype>(order, r, dw);
}

template <typename dtype> CPLocalOptimizer<dtype>::~CPLocalOptimizer() {
  // delete S;
  delete local_mttkrp;
}

template <typename dtype>
void CPLocalOptimizer<dtype>::configure(Tensor<dtype> *input,
                                        Matrix<dtype> **mat,
                                        Matrix<dtype> *grad, double lambda) {

  CPOptimizer<dtype>::configure(input, mat, grad, lambda);
  local_mttkrp->setup(input, mat);
  local_mttkrp->distribute_W();
  local_mttkrp->construct_mttkrp_locals();
  local_mttkrp->setup_V_local_data();
}

template <typename dtype> double CPLocalOptimizer<dtype>::step() {

  World *dw = this->world;
  int order = this->order;

  for (int i = 0; i < order; i++) {

    local_mttkrp->mttkrp_calc(i);

    local_mttkrp->mttkrp[i]->print();

    local_mttkrp->post_mttkrp_reduce(i);

    local_mttkrp->mttkrp[i]->print();
    break;

    CPOptimizer<dtype>::update_S(i);
    // // calculate gradient
    // this->grad_W[i]["ij"] = -M["ij"] + this->W[i]->operator[]("ik") *
    // this->S["kj"];

    cout << "in the optimizer" << i << endl;

    // subproblem M=W*S
    cholesky_solve(*(local_mttkrp->mttkrp[i]), *(this->W[i]), this->S);

    cout << "finish cholesky_solve " << i << endl;

  }
  return 1.;
}

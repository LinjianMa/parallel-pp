
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
  for (int i = 0; i < this->order; i++) {
    local_mttkrp->distribute_W(i);
  }
  local_mttkrp->construct_mttkrp_locals();
  local_mttkrp->setup_V_local_data();
}

template <typename dtype> double CPLocalOptimizer<dtype>::step() {

  World *dw = this->world;
  int order = this->order;

  for (int i = 0; i < order; i++) {

    local_mttkrp->mttkrp_calc(i);
    local_mttkrp->post_mttkrp_reduce(i);
    CPOptimizer<dtype>::update_S(i);
    // calculate gradient
    this->grad_W[i]["ij"] = -local_mttkrp->mttkrp[i]->operator[]("ij") +
                            this->W[i]->operator[]("ik") * this->S["kj"];

    // subproblem M=W*S
    Matrix<> M_reshape =
        Matrix<>(this->W[i]->nrow, this->W[i]->ncol, *(this->world));
    M_reshape["ij"] = local_mttkrp->mttkrp[i]->operator[]("ij");
    cholesky_solve(M_reshape, *(this->W[i]), this->S);

    local_mttkrp->distribute_W(i);
  }
  return 1.;
}

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
                                        Matrix<dtype> *mat, Matrix<dtype> *grad,
                                        double lambda) {

  CPOptimizer<dtype>::configure(input, mat, grad, lambda);
  local_mttkrp->setup(input, mat);
}

template <typename dtype> double CPLocalOptimizer<dtype>::step() {

  World *dw = this->world;
  int order = this->order;

  for (int i = 0; i < order; i++) {
    // make the char
    swap_char(seq_V, i, order - 1);
    /*  construct Matrix M
     *   M["dk"] = V["abcd"]*W1["ak"]*W2["bk"]*W3["ck"]
     */
    int lens_H[order];
    int index[order];
    for (int j = 0; j < order - 1; j++) {
      index[j] = (int)(seq_V[j] - 'a');
      lens_H[j] = this->W[index[j]].nrow;
    }
    index[order - 1] = (int)(seq_V[order - 1] - 'a');
    lens_H[order - 1] = this->W[i].ncol;

    local_mttkrp->distribute_mats(i);
    local_mttkrp->post_mttkrp_reduce(i);

    // initialize matrix M
    Matrix<dtype> M = Matrix<dtype>(this->W[i].nrow, this->W[i].ncol);
    // // Khatri-Rao Product C[I,J,K]= A[I,K](op)B[J,K]
    // KhatriRao_contract(M, *(this->V), this->W, index, lens_H, *dw);
    // calculating S
    CPOptimizer<dtype>::update_S(i);
    // // calculate gradient
    // this->grad_W[i]["ij"] = -M["ij"] + this->W[i]["ik"] * this->S["kj"];

    cout << "in the optimizer" << i << endl;

    // subproblem M=W*S
    cholesky_solve(M, this->W[i], this->S);

    cout << "finish cholesky_solve " << i << endl;

    // recover the char
    swap_char(seq_V, i, order - 1);
  }
  return 1.;
}

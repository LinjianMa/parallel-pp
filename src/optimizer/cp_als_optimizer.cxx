
// #include "decomposition.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPOptimizer<dtype>::CPOptimizer(int order, int r, World &dw) {

  this->world = &dw;
  this->order = order;
  this->rank = r;

  // M and S
  S = Matrix<>(r, r, *this->world);
  M = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
  WTW = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
}

template <typename dtype> CPOptimizer<dtype>::~CPOptimizer() {
  for (int i = 0; i < this->order; i++) {
    delete this->M[i];
    delete this->WTW[i];
  }
  free(this->M);
  free(this->WTW);
}

template <typename dtype> void CPOptimizer<dtype>::update_S(int update_index) {
  Timer t_hadamard_prod("hadamard_prod");
  t_hadamard_prod.start();
  S["ij"] = 1.;
  for (int i = 0; i < update_index; i++) {
    S["ij"] = S["ij"] * WTW[i]->operator[]("ij");
  }
  for (int i = update_index + 1; i < order; i++) {
    S["ij"] = S["ij"] * WTW[i]->operator[]("ij");
  }
  S["ij"] += regul["ij"];
  t_hadamard_prod.stop();
}

template <typename dtype>
void CPOptimizer<dtype>::configure(Tensor<dtype> *input, Matrix<dtype> **mat,
                                   Matrix<dtype> *grad, double lambda) {

  assert(input->order == order);

  for (int i = 0; i < order; i++) {
    assert(mat[i]->ncol == rank);
  }

  if (V != NULL) {
    delete V;
  }
  if (W != NULL) {
    delete[] this->W;
  }
  if (grad_W != NULL) {
    delete[] this->grad_W;
  }
  this->V = input;
  this->W = mat;
  this->grad_W = grad;

  for (int i = 0; i < order; i++) {
    M[i] = new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
    WTW[i] = new Matrix<>(this->rank, this->rank, *this->world);
    WTW[i]->operator[]("jk") =
        this->W[i]->operator[]("ij") * this->W[i]->operator[]("ik");
  }

  regul = Matrix<dtype>(mat[0]->ncol, mat[0]->ncol);
  regul["ii"] = 1. * lambda;
}

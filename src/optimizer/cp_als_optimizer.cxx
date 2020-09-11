
// #include "decomposition.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPOptimizer<dtype>::CPOptimizer(int order, int r, World &dw) {

  this->world = &dw;
  this->order = order;
  this->rank = r;

  // S
  S = Matrix<>(r, r);
}

template <typename dtype> CPOptimizer<dtype>::~CPOptimizer() {}

template <typename dtype> void CPOptimizer<dtype>::update_S(int update_index) {
  Timer t_hadamard_prod("hadamard_prod");
  t_hadamard_prod.start();
  // build index
  vector<int> index = vector<int>(order - 1, 0);
  int j = 0;
  for (int i = 0; i < update_index; i++) {
    index[j] = i;
    j++;
  }
  for (int i = update_index + 1; i < order; i++) {
    index[j] = i;
    j++;
  }
  // contractions
  S["ij"] = W[index[0]]->operator[]("ki") * W[index[0]]->operator[]("kj");
  for (int ii = 1; ii < order - 1; ii++) {
    S["ij"] = S["ij"] *
              (W[index[ii]]->operator[]("ki") * W[index[ii]]->operator[]("kj"));
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
  if (M != NULL) {
    delete[] this->M;
  }
  if (grad_W != NULL) {
    delete[] this->grad_W;
  }
  this->V = input;
  this->W = mat;
  this->grad_W = grad;

  M = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
  for (int i = 0; i < order; i++) {
    M[i] = new Matrix<>(this->W[i]->nrow, this->W[i]->ncol, *(this->world));
  }

  regul = Matrix<dtype>(mat[0]->ncol, mat[0]->ncol);
  regul["ii"] = 1. * lambda;
}

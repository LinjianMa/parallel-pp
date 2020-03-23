
#include "../utils/common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPPPOptimizer<dtype>::CPPPOptimizer(int order, int r, World &dw,
                                    double tol_restart_dt)
    : CPDTOptimizer<dtype>(order, r, dw) {
  this->tol_restart_dt = tol_restart_dt;
  this->W_prev = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
}

template <typename dtype> CPPPOptimizer<dtype>::~CPPPOptimizer() {
  // delete S;
}

template <typename dtype>
void CPPPOptimizer<dtype>::configure(Tensor<dtype> *input, Matrix<dtype> **mat,
                                     Matrix<dtype> *grad, double lambda) {

  CPOptimizer<dtype>::configure(input, mat, grad, lambda);
  for (int i = 0; i < this->order; i++) {
    W_prev[i] = new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
  }
}

template <typename dtype> double CPPPOptimizer<dtype>::step_dt() {

  for (int i = 0; i < this->order; i++) {
    this->W_prev[i]->operator[]("ij") = this->W[i]->operator[]("ij");
  }

  CPDTOptimizer<dtype>::step();
  CPDTOptimizer<dtype>::step();

  int num_smallupdate = 0;
  for (int i = 0; i < this->order; i++) {
    this->W_prev[i]->operator[]("ij") -= this->W[i]->operator[]("ij");
    double dW_norm = this->W_prev[i]->norm2();
    double W_norm = this->W[i]->norm2();

    if (dW_norm / W_norm < this->tol_restart_dt) {
      num_smallupdate += 1;
    }
  }

  if (num_smallupdate == this->order) {
    this->pp = true;
    this->reinitialize_tree = true;
  }

  return 1.;
}

template <typename dtype> double CPPPOptimizer<dtype>::step_pp() { return 1.; }

template <typename dtype> void CPPPOptimizer<dtype>::initialize_tree() {}

template <typename dtype> double CPPPOptimizer<dtype>::step() {

  double num_sweep = 0.;

  this->restart = false;

  if (this->pp == true) {
    if (this->reinitialize_tree == true) {
      this->restart = true;
      initialize_tree();
      this->reinitialize_tree = false;
    }
    num_sweep = step_pp();
  } else {
    num_sweep = step_dt();
  }

  return num_sweep;
}

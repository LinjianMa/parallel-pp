
#include "../utils/common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPPPLocalOptimizer<dtype>::CPPPLocalOptimizer(int order, int r, World &dw,
                                              double tol_restart_dt)
    : CPPPOptimizer<dtype>(order, r, dw, tol_restart_dt),
      CPDTLocalOptimizer<dtype>(order, r, dw, false), CPDTOptimizer<dtype>(
                                                          order, r, dw, false) {
  this->dW_local = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
  this->update_W = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
}

template <typename dtype> CPPPLocalOptimizer<dtype>::~CPPPLocalOptimizer() {
  for (int i = 0; i < this->order; i++) {
    delete this->dW_local[i];
  }
  free(this->dW_local);
}

template <typename dtype>
void CPPPLocalOptimizer<dtype>::configure(Tensor<dtype> *input,
                                          Matrix<dtype> **mat,
                                          Matrix<dtype> *grad, double lambda) {

  CPDTLocalOptimizer<dtype>::configure(input, mat, grad, lambda);
  for (int i = 0; i < this->order; i++) {
    this->dW[i] = new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
    this->update_W[i] = new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
  }
  for (int i = 0; i < this->order; i++) {
    int64_t pad_local_col =
        int(this->V->pad_edge_len[i] / this->local_mttkrp->phys_phase[i]);
    this->dW_local[i] = new Matrix<dtype>(pad_local_col, this->rank,
                                          *this->local_mttkrp->sworld);
  }
  for (int i = 0; i < this->order; i++) {
    this->local_mttkrp->distribute_W(i, this->dW, this->dW_local);
  }
}

template <typename dtype> double CPPPLocalOptimizer<dtype>::step_dt() {
  Timer t_localpp_step_dt("localpp_step_dt");
  t_localpp_step_dt.start();

  if (this->world->rank == 0) {
    cout << "***** dt step *****" << endl;
  }

  for (int i = 0; i < this->order; i++) {
    this->dW[i]->operator[]("ij") = this->W[i]->operator[]("ij");
  }

  CPDTLocalOptimizer<dtype>::step();
  CPDTLocalOptimizer<dtype>::step();

  int num_smallupdate = 0;
  for (int i = 0; i < this->order; i++) {
    this->dW[i]->operator[]("ij") -= this->W[i]->operator[]("ij");
    double dW_norm = this->dW[i]->norm2();
    double W_norm = this->W[i]->norm2();

    if (dW_norm / W_norm < this->tol_restart_dt) {
      num_smallupdate += 1;
    }
  }

  if (num_smallupdate == this->order) {
    this->pp = true;
    this->reinitialize_tree = true;
  }

  t_localpp_step_dt.stop();
  return 1.;
}

template <typename dtype> double CPPPLocalOptimizer<dtype>::step_pp() {
  Timer t_localpp_step_pp("localpp_step_pp");
  t_localpp_step_pp.start();

  if (this->world->rank == 0) {
    cout << "***** pairwise perturbation step *****" << endl;
  }

  for (int i = 0; i < this->order; i++) {
    CPOptimizer<dtype>::update_S(i);

    CPPPOptimizer<dtype>::mttkrp_approx(i, this->dW_local, this->local_mttkrp->mttkrp_local_mat[i]);
    this->local_mttkrp->post_mttkrp_reduce(i);

    this->M[i]->operator[]("ij") = this->local_mttkrp->mttkrp[i]->operator[]("ij");
    spd_solve(*this->M[i], *this->update_W[i], this->S);

    this->dW[i]->operator[]("ij") += this->update_W[i]->operator[]("ij") - this->W[i]->operator[]("ij");
    this->W[i]->operator[]("ij") = this->update_W[i]->operator[]("ij");

    this->local_mttkrp->distribute_W(i, this->local_mttkrp->W,
                                     this->local_mttkrp->W_local);
    this->local_mttkrp->distribute_W(i, this->dW, this->dW_local);
  }

  int num_bigupdate = 0;
  for (int i = 0; i < this->order; i++) {
    double dW_norm = this->dW[i]->norm2();
    double W_norm = this->W[i]->norm2();

    if (dW_norm / W_norm > this->tol_restart_dt) {
      num_bigupdate += 1;
    }
  }
  if (num_bigupdate > 0) {
    this->pp = false;
    this->reinitialize_tree = false;
  }

  t_localpp_step_pp.stop();
  return 1.;
}

template <typename dtype> double CPPPLocalOptimizer<dtype>::step() {

  double num_sweep = 0.;

  this->restart = false;

  if (this->pp == true) {
    if (this->reinitialize_tree == true) {
      this->restart = true;
      for (int i = 0; i < this->order; i++) {
        this->dW[i]->operator[]("ij") = 0.;
      }
      CPPPOptimizer<dtype>::initialize_tree(
          this->local_mttkrp->sworld, this->local_mttkrp->V_local,
          this->local_mttkrp->W_local, this->dW_local);
      this->reinitialize_tree = false;
    }
    num_sweep = step_pp();
  } else {
    num_sweep = step_dt();
  }

  return num_sweep;
}

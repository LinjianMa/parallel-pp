
#include "../utils/common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPPPLocalOptimizer<dtype>::CPPPLocalOptimizer(int order, int r, World &dw,
                                              double tol_restart_dt, bool use_msdt)
    : CPPPOptimizer<dtype>(order, r, dw, tol_restart_dt, use_msdt),
      CPDTLocalOptimizer<dtype>(order, r, dw, use_msdt), CPDTOptimizer<dtype>(
                                                          order, r, dw, use_msdt) {
  this->dW_local = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
  this->WTW_local = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
  this->WTdW_local = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
}

template <typename dtype> CPPPLocalOptimizer<dtype>::~CPPPLocalOptimizer() {
  for (int i = 0; i < this->order; i++) {
    delete this->dW_local[i];
    delete this->WTW_local[i];
    delete this->WTdW_local[i];
  }
  free(this->dW_local);
  free(this->WTW_local);
  free(this->WTdW_local);
}

template <typename dtype>
void CPPPLocalOptimizer<dtype>::configure(Tensor<dtype> *input,
                                          Matrix<dtype> **mat,
                                          Matrix<dtype> *grad, double lambda) {

  CPDTLocalOptimizer<dtype>::configure(input, mat, grad, lambda);
  for (int i = 0; i < this->order; i++) {
    this->dW[i] = new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
    this->update_W[i] =
        new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
    this->WTdW[i] = new Matrix<>(this->rank, this->rank, *this->world);
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
  // S_local, S_local_temp, WTW_local, WTdW_local
  int phys_phase[2];
  phys_phase[0] = this->S.edge_map[0].calc_phys_phase();
  phys_phase[1] = this->S.edge_map[1].calc_phys_phase();
  for (int i = 0; i < this->order; i++) {
    IASSERT(phys_phase[0] == this->WTW[i]->edge_map[0].calc_phys_phase());
    IASSERT(phys_phase[1] == this->WTW[i]->edge_map[1].calc_phys_phase());
    IASSERT(phys_phase[0] == this->WTdW[i]->edge_map[0].calc_phys_phase());
    IASSERT(phys_phase[1] == this->WTdW[i]->edge_map[1].calc_phys_phase());
  }
  int64_t pad_row = int(this->S.pad_edge_len[0] / phys_phase[0]);
  int64_t pad_col = int(this->S.pad_edge_len[1] / phys_phase[1]);
  int64_t num_elements = pad_row * pad_col;

  S_local = Matrix<>(pad_row, pad_col, *this->local_mttkrp->sworld);
  memcpy(S_local.data, this->S.data, sizeof(dtype) * num_elements);

  S_local_temp = Matrix<>(pad_row, pad_col, *this->local_mttkrp->sworld);
  for (int i = 0; i < this->order; i++) {
    WTW_local[i] = new Matrix<>(pad_row, pad_col, *this->local_mttkrp->sworld);
    WTdW_local[i] = new Matrix<>(pad_row, pad_col, *this->local_mttkrp->sworld);

    memcpy(WTW_local[i]->data, this->WTW[i]->data,
           sizeof(dtype) * num_elements);
    memcpy(WTdW_local[i]->data, this->WTdW[i]->data,
           sizeof(dtype) * num_elements);
  }

  if (this->use_msdt == false) {
    this->ppdt = new PPDimensionTree(this->order, this->world, input);
  } else {
    this->ppdt = new PPDimensionTree(this->order, this->world, input,
                                     this->local_mttkrp->trans_V_local_map,
                                     this->local_mttkrp->trans_V_str_map);
  }
}

template <typename dtype> double CPPPLocalOptimizer<dtype>::step_dt() {
  Timer t_localpp_step_dt("localpp_step_dt");
  t_localpp_step_dt.start();
  double num_sweep = 0.;

  if (this->world->rank == 0) {
    cout << "***** dt step *****" << endl;
  }

  for (int i = 0; i < this->order; i++) {
    this->dW[i]->operator[]("ij") = this->W[i]->operator[]("ij");
  }

  if (this->use_msdt == false) {
    CPDTLocalOptimizer<dtype>::step();
    CPDTLocalOptimizer<dtype>::step();
    num_sweep = 1.;
  } else {
    CPDTLocalOptimizer<dtype>::step();
    num_sweep = 1. * (this->order - 1) / this->order;
  }

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
  return num_sweep;
}

template <typename dtype>
void CPPPLocalOptimizer<dtype>::pp_update_after_solve(int i) {
  Timer t_localpp_step_pp("pp_update_after_solve");
  t_localpp_step_pp.start();

  this->WTW[i]->operator[]("jk") =
      this->update_W[i]->operator[]("ij") * this->update_W[i]->operator[]("ik");

  // TODO: these two lines can be faster
  this->dW[i]->operator[]("ij") +=
      this->update_W[i]->operator[]("ij") - this->W[i]->operator[]("ij");
  this->W[i]->operator[]("ij") = this->update_W[i]->operator[]("ij");

  this->WTdW[i]->operator[]("jk") =
      this->W[i]->operator[]("ij") * this->dW[i]->operator[]("ik");
  memcpy(WTW_local[i]->data, this->WTW[i]->data,
         sizeof(dtype) * WTW_local[i]->ncol * WTW_local[i]->nrow);
  memcpy(WTdW_local[i]->data, this->WTdW[i]->data,
         sizeof(dtype) * WTdW_local[i]->ncol * WTdW_local[i]->nrow);
  t_localpp_step_pp.stop();
}

template <typename dtype> double CPPPLocalOptimizer<dtype>::step_pp() {
  Timer t_localpp_step_pp("localpp_step_pp");
  t_localpp_step_pp.start();

  if (this->world->rank == 0) {
    cout << "***** pairwise perturbation step *****" << endl;
  }

  for (int i = 0; i < this->order; i++) {
    CPPPOptimizer<dtype>::mttkrp_approx(
        i, this->dW_local, this->local_mttkrp->mttkrp_local_mat[i]);
    this->local_mttkrp->post_mttkrp_reduce(i);
    this->M[i]->operator[]("ij") =
        this->local_mttkrp->mttkrp[i]->operator[]("ij");
    // second order correction
    CPPPOptimizer<dtype>::mttkrp_approx_second_correction(
        i, this->S_local, this->S_local_temp, this->WTW_local,
        this->WTdW_local);
    memcpy(this->S.data, this->S_local.data,
           sizeof(dtype) * S_local.ncol * S_local.nrow);
    this->M[i]->operator[]("ij") +=
        this->W[i]->operator[]("ik") * this->S["kj"];

    CPOptimizer<dtype>::update_S(i);
    spd_solve(*this->M[i], *this->update_W[i], this->S);
    pp_update_after_solve(i);

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
        this->dW_local[i]->operator[]("ij") = 0.;
        this->WTdW[i]->operator[]("ij") = 0.;
      }
      this->ppdt->initialize_tree(this->local_mttkrp->sworld,
                                  this->local_mttkrp->V_local,
                                  this->local_mttkrp->W_local);
      this->reinitialize_tree = false;
      for (int i = 0; i < this->order; i++) {
        memcpy(WTW_local[i]->data, this->WTW[i]->data,
               sizeof(dtype) * WTW_local[i]->ncol * WTW_local[i]->nrow);
        memcpy(WTdW_local[i]->data, this->WTdW[i]->data,
               sizeof(dtype) * WTdW_local[i]->ncol * WTdW_local[i]->nrow);
      }
    }
    num_sweep = step_pp();
  } else {
    num_sweep = step_dt();
  }

  return num_sweep;
}

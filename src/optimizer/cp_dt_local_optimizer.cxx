
#include "../utils/common.h"
#include "../utils/dimension_tree.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPDTLocalOptimizer<dtype>::CPDTLocalOptimizer(int order, int r, World &dw,
                                              bool use_msdt)
    : CPDTOptimizer<dtype>(order, r, dw, use_msdt) {

  local_mttkrp = new LocalMTTKRP<dtype>(order, r, dw);
}

template <typename dtype> CPDTLocalOptimizer<dtype>::~CPDTLocalOptimizer() {
  // delete S;
  delete local_mttkrp;
}

template <typename dtype>
void CPDTLocalOptimizer<dtype>::configure(Tensor<dtype> *input,
                                          Matrix<dtype> **mat,
                                          Matrix<dtype> *grad, double lambda) {

  CPOptimizer<dtype>::configure(input, mat, grad, lambda);
  local_mttkrp->setup(input, mat);
  for (int i = 0; i < this->order; i++) {
    local_mttkrp->distribute_W(i, local_mttkrp->W, local_mttkrp->W_local);
  }
  local_mttkrp->construct_mttkrp_locals();
  local_mttkrp->setup_V_local_data();
}

template <typename dtype>
void CPDTLocalOptimizer<dtype>::solve_one_mode(int i) {
  int ii = this->indexes[i];
  vector<int> mat_index = {i};

  string mat_seq;
  vec2str(mat_index, mat_seq);

  if (this->mttkrp_map.find(mat_seq) == this->mttkrp_map.end()) {
    CPDTOptimizer<dtype>::mttkrp_map_DT(mat_seq, local_mttkrp->sworld,
                                        local_mttkrp->W_local,
                                        local_mttkrp->V_local);
  }
  local_mttkrp->mttkrp_local_mat[ii]->operator[]("ij") =
      this->mttkrp_map[mat_seq]->operator[]("ij");

  local_mttkrp->post_mttkrp_reduce(ii);

  // calculating S
  CPOptimizer<dtype>::update_S(ii);
  // calculate gradient
  this->grad_W[ii]["ij"] = -local_mttkrp->mttkrp[ii]->operator[]("ij") +
                           this->W[ii]->operator[]("ik") * this->S["kj"];

  Matrix<> M_reshape =
      Matrix<>(this->W[ii]->nrow, this->W[ii]->ncol, *(this->world));
  M_reshape["ij"] = local_mttkrp->mttkrp[ii]->operator[]("ij");
  spd_solve(M_reshape, *this->W[ii], this->S);

  local_mttkrp->distribute_W(ii, local_mttkrp->W, local_mttkrp->W_local);
}

template <typename dtype> double CPDTLocalOptimizer<dtype>::step_dt() {

  if (this->first_subtree) {
    this->indexes = this->indexes1;
    this->left_index = this->left_index1;
  } else {
    this->indexes = this->indexes2;
    this->left_index = this->left_index2;
  }
  // clear the Hash Table
  for (auto const &x : this->mttkrp_map) {
    delete x.second;
  }
  this->mttkrp_map.clear();
  // reinitialize
  CPDTOptimizer<dtype>::mttkrp_map_init(this->left_index, local_mttkrp->sworld,
                                        local_mttkrp->W_local,
                                        local_mttkrp->V_local);

  // iteration on W[i]
  for (int i = 0; i < this->indexes.size(); i++) {
    if (!this->first_subtree && this->indexes[i] != this->special_index)
      continue;
    solve_one_mode(i);
  }

  this->first_subtree = !this->first_subtree;
  return 0.5;
}

template <typename dtype> double CPDTLocalOptimizer<dtype>::step_msdt() {

  // clear the Hash Table
  for (auto const &x : this->mttkrp_map) {
    delete x.second;
  }
  this->mttkrp_map.clear();

  // reinitialize
  this->dt->update_indexes(this->indexes, this->left_index);
  CPDTOptimizer<dtype>::mttkrp_map_init(this->left_index, local_mttkrp->sworld,
                                        local_mttkrp->W_local,
                                        local_mttkrp->V_local);

  // iteration on W[i]
  for (int i = 0; i < this->indexes.size(); i++) {
    solve_one_mode(i);
  }

  this->left_index = (this->left_index + this->order - 1) % this->order;
  return 1. * (this->order - 1) / this->order;
}

template <typename dtype> double CPDTLocalOptimizer<dtype>::step() {
  if (this->use_msdt == true) {
    return step_msdt();
  } else {
    return step_dt();
  }
}
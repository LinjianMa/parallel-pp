
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

template <typename dtype>
CPDTLocalOptimizer<dtype>::CPDTLocalOptimizer(int order, int r, World &dw,
                                              bool use_msdt,
                                              bool renew_ppoperator)
    : CPDTOptimizer<dtype>(order, r, dw, use_msdt, renew_ppoperator) {
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
  if (this->use_msdt == true) {
    local_mttkrp->get_V_local_transposes();
  }

  this->is_equidimentional = true;
  for (int i = 1; i < this->order; i++) {
    if (this->local_mttkrp->V_local->lens[i] !=
        this->local_mttkrp->V_local->lens[0]) {
      this->is_equidimentional = false;
      break;
    }
  }
  if (this->world->rank == 0) {
    cout << "is equidimentional: " << this->is_equidimentional << endl;
    for (int i = 0; i < this->order; i++) {
      cout << "V_local_lens[" << i
           << "]: " << this->local_mttkrp->V_local->lens[i] << endl;
    }
  }
}

template <typename dtype>
void CPDTLocalOptimizer<dtype>::solve_one_mode(int i) {
  Timer t_DT_solve_one_mode("DT_solve_one_mode");
  t_DT_solve_one_mode.start();
  int ii = this->indexes[i];
  vector<int> mat_index = {i};

  string mat_seq;
  vec2str(mat_index, mat_seq);

  if (this->mttkrp_exist_map.find(mat_seq) == this->mttkrp_exist_map.end()) {
    Timer t_mttkrp_map_DT("multi-TTV");
    t_mttkrp_map_DT.start();
    CPDTOptimizer<dtype>::mttkrp_map_DT(mat_seq, local_mttkrp->sworld,
                                        local_mttkrp->W_local,
                                        local_mttkrp->V_local);
    t_mttkrp_map_DT.stop();
  }
  local_mttkrp->mttkrp_local_mat[ii]->operator[]("ij") =
      this->mttkrp_map[mat_seq]->operator[]("ij");

  local_mttkrp->post_mttkrp_reduce(ii);

  // calculating S
  CPOptimizer<dtype>::update_S(ii);
  // calculate gradient
  this->grad_W[ii]["ij"] = -local_mttkrp->mttkrp[ii]->operator[]("ij") +
                           this->W[ii]->operator[]("ik") * this->S["kj"];

  this->M[ii]->operator[]("ij") = local_mttkrp->mttkrp[ii]->operator[]("ij");
  spd_solve(*this->M[ii], *this->W[ii], this->S);
  this->WTW[ii]->operator[]("jk") =
      this->W[ii]->operator[]("ij") * this->W[ii]->operator[]("ik");

  local_mttkrp->distribute_W(ii, local_mttkrp->W, local_mttkrp->W_local);
  t_DT_solve_one_mode.stop();
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
  if (this->is_equidimentional == false) {
    for (auto const &x : this->mttkrp_map) {
      delete x.second;
    }
    this->mttkrp_map.clear();
  }
  this->mttkrp_exist_map.clear();
  // reinitialize
  CPDTOptimizer<dtype>::mttkrp_map_init(
      this->left_index, local_mttkrp->sworld, local_mttkrp->W_local,
      local_mttkrp->V_local, this->seq_V, local_mttkrp->V_local->lens);

  // iteration on W[i]
  for (int i = 0; i < this->indexes.size(); i++) {
    if (!this->first_subtree && this->indexes[i] != this->special_index)
      continue;
    solve_one_mode(i);
  }

  this->first_subtree = !this->first_subtree;
  return 0.5;
}

template <typename dtype> double CPDTLocalOptimizer<dtype>::step_msdt_specific_subtree(int left_index) {
  Timer t_step_msdt("step_msdt");
  t_step_msdt.start();
  // clear the Hash Table
  if (this->is_equidimentional == false) {
    for (auto const &x : this->mttkrp_map) {
      if (find(this->inter_for_pp.begin(), this->inter_for_pp.end(),
               x.second) == this->inter_for_pp.end()) {
        delete x.second;
      }
    }
    this->mttkrp_map.clear();
  }
  this->mttkrp_exist_map.clear();

  // consider init_pp
  if (this->renew_ppoperator == true) {
    CPDTOptimizer<dtype>::construct_inter_for_pp(
        local_mttkrp->sworld, local_mttkrp->V_local->lens, left_index);
    this->mttkrp_map[this->seq_tree_top] =
        this->inter_for_pp[this->inter_for_pp.size() - 1];
  }

  // reinitialize
  this->dt->update_indexes(this->indexes, left_index);
  CPDTOptimizer<dtype>::mttkrp_map_init(
      left_index, local_mttkrp->sworld, local_mttkrp->W_local,
      local_mttkrp->trans_V_local_map[left_index],
      local_mttkrp->trans_V_str_map[left_index].c_str(),
      local_mttkrp->V_local->lens);

  // iteration on W[i]
  for (int i = 0; i < this->indexes.size(); i++) {
    solve_one_mode(i);
  }
  t_step_msdt.stop();
  return 1. * (this->order - 1) / this->order;
}

template <typename dtype> double CPDTLocalOptimizer<dtype>::step_msdt() {
  double sweeps = step_msdt_specific_subtree(this->left_index);
  this->left_index = (this->left_index + this->order - 1) % this->order;
  return sweeps;
}

template <typename dtype> double CPDTLocalOptimizer<dtype>::step() {
  if (this->use_msdt == true) {
    return step_msdt();
  } else {
    return step_dt();
  }
}
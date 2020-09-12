
#include "../utils/common.h"
#include "../utils/dimension_tree.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPDTOptimizer<dtype>::CPDTOptimizer(int order, int r, World &dw, bool use_msdt)
    : CPOptimizer<dtype>(order, r, dw) {

  this->use_msdt = use_msdt;
  dt = new DimensionTree(order);

  // make the char seq_V
  seq_V[order] = '\0';
  seq_tree_top[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
    seq_tree_top[j] = seq_V[j];
  }
  seq_tree_top[order - 1] = '*';

  // initialize the indexes
  indexes = vector<int>(order - 1, 0);
  for (int i = 0; i < indexes.size(); i++) {
    indexes[i] = i;
  }

  indexes1 = indexes;
  indexes2 = indexes1;

  left_index = order - 1;
  left_index1 = left_index;
  left_index2 = 0; // (left_index + order - 1) % order;
  dt->update_indexes(indexes2, left_index2);
  special_index = order - 1;

  first_subtree = true;
}

template <typename dtype> CPDTOptimizer<dtype>::~CPDTOptimizer() {
  // delete S;
}

template <typename dtype>
void CPDTOptimizer<dtype>::configure(Tensor<dtype> *input, Matrix<dtype> **mat,
                                     Matrix<dtype> *grad, double lambda) {

  CPOptimizer<dtype>::configure(input, mat, grad, lambda);
  this->is_equidimentional = true;
  for (int i = 1; i < this->order; i++) {
    if (this->V->lens[i] != this->V->lens[0]) {
      this->is_equidimentional = false;
      break;
    }
  }
}

template <typename dtype>
void CPDTOptimizer<dtype>::mttkrp_map_init(int left_index, World *dw,
                                           Matrix<> **mat, Tensor<> *T,
                                           const char *seq_T,
                                           int64_t *init_tensor_lens) {
  Timer t_mttkrp_map_init("mttkrp_map_init");
  t_mttkrp_map_init.start();

  int order = this->order;

  // build seq_map_init
  seq_map_init[order] = '\0';
  seq_map_init[order - 1] = '*';
  int j = 0;
  for (int i = left_index + 1; i < order; i++) {
    seq_map_init[j] = 'a' + i;
    j++;
  }
  for (int i = 0; i < left_index; i++) {
    seq_map_init[j] = 'a' + i;
    j++;
  }
  // build seq_matrix
  char seq_matrix[3];
  seq_matrix[2] = '\0';
  seq_matrix[1] = '*';
  seq_matrix[0] = 'a' + left_index;
  // store that into the mttkrp_map
  int lens[strlen(seq_map_init)];
  for (int ii = 0; ii < strlen(seq_map_init); ii++) {
    if (seq_map_init[ii] == '*')
      lens[ii] = mat[0]->ncol;
    else
      lens[ii] = init_tensor_lens[int(seq_map_init[ii] - 'a')];
  }
  Timer t_mttkrp_map_first_intermediate(
      "mttkrp_map_first_intermediate");
  t_mttkrp_map_first_intermediate.start();
  if (mttkrp_map.find(seq_tree_top) == mttkrp_map.end()) {
    mttkrp_map[seq_tree_top] = new Tensor<dtype>(strlen(seq_map_init), lens, *dw);
    mttkrp_map[seq_tree_top]->operator[](seq_map_init) +=
        (*T)[seq_T] * mat[left_index]->operator[](seq_matrix);
  } else {
    mttkrp_map[seq_tree_top]->operator[](seq_map_init) =
        (*T)[seq_T] * mat[left_index]->operator[](seq_matrix);
  }
  mttkrp_exist_map[seq_tree_top] = true;
  t_mttkrp_map_first_intermediate.stop();

  t_mttkrp_map_init.stop();
}

template <typename dtype>
void CPDTOptimizer<dtype>::mttkrp_map_DT(string index, World *dw,
                                         Matrix<> **mat, Tensor<> *T) {
  Timer t_mttkrp_map_DT("mttkrp_map_DT");
  t_mttkrp_map_DT.start();

  char const *index_char = index.c_str();

  char const *parent_index = dt->parent[index].c_str();
  if (mttkrp_exist_map.find(parent_index) == mttkrp_exist_map.end()) {
    mttkrp_map_DT(parent_index, dw, mat, T);
  }
  // get the modindexe of mat
  char const *mat_index = dt->contract_index[index].c_str();
  int W_index = int(mat_index[0] - 'a');
  int lens[strlen(index_char)];

  for (int ii = 0; ii < strlen(index_char); ii++) {
    if (index[ii] == '*')
      lens[ii] = mat[0]->ncol;
    else
      lens[ii] = T->lens[int(indexes[index[ii] - 'a'])];
  }
  if (mttkrp_map.find(index) == mttkrp_map.end()) {
    mttkrp_map[index] = new Tensor<dtype>(strlen(index_char), lens, *dw);
    mttkrp_map[index]->operator[](index_char) +=
        mttkrp_map[parent_index]->operator[](parent_index) *
        mat[indexes[W_index]]->operator[](mat_index);
  } else {
    mttkrp_map[index]->operator[](index_char) =
        mttkrp_map[parent_index]->operator[](parent_index) *
        mat[indexes[W_index]]->operator[](mat_index);
  }
  mttkrp_exist_map[index] = true;

  t_mttkrp_map_DT.stop();
}

template <typename dtype> void CPDTOptimizer<dtype>::solve_one_mode(int i) {
  vector<int> mat_index = {i};
  int ii = indexes[i];

  string mat_seq;
  vec2str(mat_index, mat_seq);

  if (mttkrp_exist_map.find(mat_seq) == mttkrp_exist_map.end()) {
    mttkrp_map_DT(mat_seq, this->world, this->W, this->V);
  }
  this->M[ii]->operator[]("ij") = mttkrp_map[mat_seq]->operator[]("ij");

  // calculating S
  CPOptimizer<dtype>::update_S(ii);
  // calculate gradient
  this->grad_W[ii]["ij"] =
      -this->M[ii]->operator[]("ij") + this->W[ii]->operator[]("ik") * this->S["kj"];

  spd_solve(*this->M[ii], *this->W[ii], this->S);
}

template <typename dtype> double CPDTOptimizer<dtype>::step_dt() {

  if (first_subtree) {
    indexes = indexes1;
    left_index = left_index1;
  } else {
    indexes = indexes2;
    left_index = left_index2;
  }
  // clear the Hash Table
  if (this->is_equidimentional == false) {
    for (auto const &x : this->mttkrp_map) {
      delete x.second;
    }
    mttkrp_map.clear();
  }
  mttkrp_exist_map.clear();

  // reinitialize
  mttkrp_map_init(left_index, this->world, this->W, this->V, this->seq_V,
                  this->V->lens);

  // iteration on W[i]
  for (int i = 0; i < indexes.size(); i++) {
    if (!first_subtree && indexes[i] != special_index)
      continue;
    solve_one_mode(i);
  }

  first_subtree = !first_subtree;

  return 0.5;
}

template <typename dtype> double CPDTOptimizer<dtype>::step_msdt() {

  // clear the Hash Table
  if (this->is_equidimentional == false) {
    for (auto const &x : this->mttkrp_map) {
      delete x.second;
    }
    mttkrp_map.clear();
  }
  mttkrp_exist_map.clear();

  // reinitialize
  dt->update_indexes(indexes, left_index);
  mttkrp_map_init(left_index, this->world, this->W, this->V, this->seq_V,
                  this->V->lens);

  // iteration on W[i]
  for (int i = 0; i < indexes.size(); i++) {
    solve_one_mode(i);
  }

  left_index = (left_index + this->order - 1) % this->order;

  return 1. * (this->order - 1) / this->order;
}

template <typename dtype> double CPDTOptimizer<dtype>::step() {
  if (this->use_msdt == true) {
    return step_msdt();
  } else {
    return step_dt();
  }
}

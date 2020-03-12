
#include "../utils/common.h"
#include "../utils/dimension_tree.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPDTOptimizer<dtype>::CPDTOptimizer(int order, int r, World &dw)
    : CPOptimizer<dtype>(order, r, dw) {

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
  left_index2 = (left_index + order - 1) % order;
  dt->update_indexes(indexes2, left_index2);
  special_index = 0;

  first_subtree = true;
}

template <typename dtype> CPDTOptimizer<dtype>::~CPDTOptimizer() {
  // delete S;
}

template <typename dtype>
void CPDTOptimizer<dtype>::mttkrp_map_init(int left_index) {
  World *dw = this->world;
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
      lens[ii] = this->W[0].ncol;
    else
      lens[ii] = this->V->lens[int(seq_map_init[ii] - 'a')];
  }
  mttkrp_map[seq_tree_top] = Tensor<dtype>(strlen(seq_map_init), lens, *dw);

  mttkrp_map[seq_tree_top][seq_map_init] =
      (*this->V)[seq_V] * this->W[left_index][seq_matrix];
}

template <typename dtype>
void CPDTOptimizer<dtype>::mttkrp_map_DT(string index) {
  World *dw = this->world;
  char const *index_char = index.c_str();

  char const *parent_index = dt->parent[index].c_str();
  if (mttkrp_map.find(parent_index) == mttkrp_map.end()) {
    mttkrp_map_DT(parent_index);
  }
  // get the modindexe of W
  char const *mat_index = dt->contract_index[index].c_str();
  int W_index = int(mat_index[0] - 'a');
  int lens[strlen(index_char)];

  for (int ii = 0; ii < strlen(index_char); ii++) {
    if (index[ii] == '*')
      lens[ii] = this->W[0].ncol;
    else
      lens[ii] = this->V->lens[int(indexes[index[ii] - 'a'])];
  }
  mttkrp_map[index] = Tensor<dtype>(strlen(index_char), lens, *dw);

  // TODO: this needs to be implemented with a local version.
  mttkrp_map[index][index_char] = mttkrp_map[parent_index][parent_index] *
                                  this->W[indexes[W_index]][mat_index];
}

template <typename dtype> double CPDTOptimizer<dtype>::step() {

  World *dw = this->world;
  int order = this->order;

  if (first_subtree) {
    indexes = indexes1;
    left_index = left_index1;
  } else {
    indexes = indexes2;
    left_index = left_index2;
  }
  // clear the Hash Table
  mttkrp_map.clear();

  // reinitialize
  mttkrp_map_init(left_index);

  // iteration on W[i]
  for (int i = 0; i < indexes.size(); i++) {

    if (!first_subtree && i > special_index)
      break;
    /*  construct Matrix M
     *   M["dk"] = V["abcd"]*W1["ak"]*W2["bk"]*W3["ck"]
     */
    vector<int> mat_index = {i};

    string mat_seq;
    vec2str(mat_index, mat_seq);

    if (mttkrp_map.find(mat_seq) == mttkrp_map.end()) {
      mttkrp_map_DT(mat_seq);
    }
    Matrix<dtype> M = mttkrp_map[mat_seq];

    // calculating S
    CPOptimizer<dtype>::update_S(indexes[i]);
    // calculate gradient
    this->grad_W[indexes[i]]["ij"] =
        -M["ij"] + this->W[indexes[i]]["ik"] * this->S["kj"];

    cholesky_solve(M, this->W[indexes[i]], this->S);
  }

  first_subtree = !first_subtree;

  return 0.5;
}

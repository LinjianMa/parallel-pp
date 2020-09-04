
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
void CPDTOptimizer<dtype>::mttkrp_map_init(int left_index, World *dw,
                                           Matrix<> **mat, Tensor<> *T) {
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
      lens[ii] = T->lens[int(seq_map_init[ii] - 'a')];
  }
  Timer t_mttkrp_map_first_intermediate_init("mttkrp_map_first_intermediate_init");
  t_mttkrp_map_first_intermediate_init.start();
  mttkrp_map[seq_tree_top] = new Tensor<dtype>(strlen(seq_map_init), lens, *dw);
  t_mttkrp_map_first_intermediate_init.stop();

  Timer t_mttkrp_map_first_intermediate("mttkrp_map_first_intermediate");
  t_mttkrp_map_first_intermediate.start();
  mttkrp_map[seq_tree_top]->operator[](seq_map_init) =
      (*T)[seq_V] * mat[left_index]->operator[](seq_matrix);
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
  if (mttkrp_map.find(parent_index) == mttkrp_map.end()) {
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
  mttkrp_map[index] = new Tensor<dtype>(strlen(index_char), lens, *dw);

  mttkrp_map[index]->operator[](index_char) = mttkrp_map[parent_index]->operator[](parent_index) *
                                  mat[indexes[W_index]]->operator[](mat_index);

  t_mttkrp_map_DT.stop();
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
  for (auto const& x : this->mttkrp_map) {
    delete x.second;
  }
  mttkrp_map.clear();

  // reinitialize
  mttkrp_map_init(left_index, this->world, this->W, this->V);

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
      mttkrp_map_DT(mat_seq, this->world, this->W, this->V);
    }
    Matrix<dtype> M = * mttkrp_map[mat_seq];

    // calculating S
    CPOptimizer<dtype>::update_S(indexes[i]);
    // calculate gradient
    this->grad_W[indexes[i]]["ij"] =
        -M["ij"] + this->W[indexes[i]]->operator[]("ik") * this->S["kj"];

    cholesky_solve(M, *this->W[indexes[i]], this->S);
  }

  first_subtree = !first_subtree;

  return 0.5;
}

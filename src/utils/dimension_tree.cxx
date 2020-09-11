
#include "dimension_tree.h"
#include "common.h"
#include <ctf.hpp>

using namespace CTF;

DimensionTree::DimensionTree(int order) {

  this->order = order;
  // construct the tree
  Construct_Dimension_Tree();
}

DimensionTree::~DimensionTree() {}

void DimensionTree::update_indexes(vector<int> &indexes, int left_index) {
  int j = 0;
  for (int i = left_index + 1; i < this->order; i++) {
    indexes[j] = i;
    j++;
  }
  for (int i = 0; i < left_index; i++) {
    indexes[j] = i;
    j++;
  }
}

void DimensionTree::Construct_Dimension_Tree() {
  int order = this->order;
  vector<int> top_node = vector<int>(order - 1);
  for (int i = 0; i < top_node.size(); i++) {
    top_node[i] = i;
  }

  Construct_Subtree(top_node);
}

void DimensionTree::Construct_Subtree(vector<int> top_node) {
  Right_Subtree(top_node);

  vector<int> child_node = vector<int>(top_node.size() - 1);
  for (int i = 0; i < child_node.size(); i++) {
    child_node[i] = top_node[i];
  }

  vector<int> mat_index = {top_node[top_node.size() - 1]};

  string child_seq, top_seq, mat_seq;
  vec2str(child_node, child_seq);
  vec2str(top_node, top_seq);
  vec2str(mat_index, mat_seq);

  parent[child_seq] = top_seq;
  contract_index[child_seq] = mat_seq;

  if (child_node.size() > 1) {
    Construct_Subtree(child_node);
  }
}

void DimensionTree::Right_Subtree(vector<int> top_node) {
  // construct the right tree
  vector<int> child_node = vector<int>(top_node.size() - 1);
  for (int i = 0; i < child_node.size(); i++) {
    child_node[i] = top_node[i + 1];
  }
  vector<int> mat_index = {top_node[0]};

  string child_seq, top_seq, mat_seq;
  vec2str(child_node, child_seq);
  vec2str(top_node, top_seq);
  vec2str(mat_index, mat_seq);

  parent[child_seq] = top_seq;
  contract_index[child_seq] = mat_seq;

  if (child_node.size() > 1) {
    Right_Subtree(child_node);
  }
}

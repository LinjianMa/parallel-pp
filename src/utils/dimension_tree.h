#ifndef __DIMENSION_TREE_H__
#define __DIMENSION_TREE_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

class DimensionTree {

public:
  DimensionTree(int order);

  ~DimensionTree();

  /**
   * \brief Update the indexes for the contraction.
   */
  void update_indexes(vector<int> &indexes, int left_index);

  /**
   * \brief Construct the dimension tree.
   * Note that this function will not construct the first level trees.
   */
  void Construct_Dimension_Tree();

  void Construct_Subtree(vector<int> top_node);

  void Right_Subtree(vector<int> top_node);

  int order;

  char seq_V[100];
  // used for doing the first contraction
  // mttkrp_map[seq][seq_map_init] = V[seq_V] * W[i][seq2]
  char seq_map_init[100];
  // used for building the MSDT.
  // sub of seq_V
  char seq_tree_top[100];

  // maps
  map<string, string> parent;
  map<string, string> contract_index;
};

#endif

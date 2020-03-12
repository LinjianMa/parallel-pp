#ifndef __CP_DT_OPTIMIZER_H__
#define __CP_DT_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class CPDTOptimizer : public CPOptimizer<dtype> {

public:
  CPDTOptimizer(int order, int r, World &dw);

  ~CPDTOptimizer();

  double step();

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

  /**
   * \brief First level MTTKRP contractions.
   */
  void mttkrp_map_init(int left_index);

  /**
   * \brief MTTKRP contractions except the first level.
   */
  void mttkrp_map_DT(string index);

  char seq_V[100];
  // used for doing the first contraction
  // mttkrp_map[seq][seq_map_init] = V[seq_V] * W[i][seq2]
  char seq_map_init[100];
  // used for building the MSDT.
  // sub of seq_V
  char seq_tree_top[100];

  // maps
  map<string, Tensor<dtype>> mttkrp_map;
  map<string, string> parent;
  map<string, string> contract_index;

  // indices that update in one step
  bool first_subtree;
  // The indexes to calculate MTTKRP wrt
  vector<int> indexes;
  // indices from left dimension tree
  vector<int> indexes1;
  // indices from the right tree
  vector<int> indexes2;
  int left_index;
  // two index that the first contraction is made wrt
  // M[str] = V[str] * W[left_index1][str]
  int left_index1;
  int left_index2;
  int special_index;
};

#include "cp_dt_optimizer.cxx"

#endif

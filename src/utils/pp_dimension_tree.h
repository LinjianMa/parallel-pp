#ifndef __PP_DIMENSION_TREE_H__
#define __PP_DIMENSION_TREE_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

class PPDimensionTree {

public:
  PPDimensionTree(int order, World *world, Tensor<> *T);

  PPDimensionTree(int order, World *world, Tensor<> *T,
                  map<int, Tensor<> *> trans_T_map,
                  map<int, string> trans_T_str_map);

  ~PPDimensionTree();

  vector<string> get_einstr(vector<int> nodeindex, vector<int> parent_nodeindex,
                            int contract_index);

  string get_nodename(vector<int> nodeindex);

  void get_parentnode(vector<int> nodeindex, string &parent_nodename,
                      vector<int> &parent_index, int &contract_index);

  void initialize_treenode(vector<int> nodeindex, Matrix<> **mat);

  void initialize_tree_root();

  void initialize_tree(Matrix<> **mat);

  int order;
  World *world;

  map<string, Tensor<> *> name_tensor_map;
  map<string, vector<int>> name_index_map;

  Tensor<> *T = NULL;
  // The map storing the transposes of the input tensor (used in MSDT).
  map<int, Tensor<> *> trans_T_map;
  map<int, string> trans_T_str_map;
  bool use_transpose_T = false;

  vector<int> fulllist = {};
  World dw;
};

#endif

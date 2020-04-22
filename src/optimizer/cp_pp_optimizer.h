#ifndef __CP_PP_OPTIMIZER_H__
#define __CP_PP_OPTIMIZER_H__

#include "../utils/dimension_tree.h"
#include <ctf.hpp>
#include <fstream>

using namespace CTF;

template <typename dtype> class CPPPOptimizer : public CPDTOptimizer<dtype> {

public:
  CPPPOptimizer(int order, int r, World &dw, double tol_restart_dt);

  ~CPPPOptimizer();

  double step();

  double step_dt();

  double step_pp();

  void configure(Tensor<dtype> *input, Matrix<dtype> **mat, Matrix<dtype> *grad,
                 double lambda);

  vector<string> get_einstr(vector<int> nodeindex, vector<int> parent_nodeindex,
                            int contract_index);

  string get_nodename(vector<int> nodeindex);

  void get_parentnode(vector<int> nodeindex, string &parent_nodename,
                      vector<int> &parent_index, int &contract_index);

  void initialize_treenode(vector<int> nodeindex);

  void initialize_tree();

  double tol_restart_dt;
  bool restart;
  bool pp = false;
  bool reinitialize_tree;

  map<string, Tensor<dtype> *> name_tensor_map;
  map<string, vector<int>> name_index_map;

  Matrix<> **dW;
};

#include "cp_pp_optimizer.cxx"

#endif
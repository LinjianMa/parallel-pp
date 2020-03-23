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

  void initialize_tree();

  void configure(Tensor<dtype> *input, Matrix<dtype> **mat, Matrix<dtype> *grad,
                 double lambda);

  double tol_restart_dt;
  bool restart;
  bool pp = false;
  bool reinitialize_tree;

  Matrix<> **W_prev;
};

#include "cp_pp_optimizer.cxx"

#endif

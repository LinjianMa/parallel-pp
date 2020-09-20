#ifndef __CP_PP_OPTIMIZER_H__
#define __CP_PP_OPTIMIZER_H__

#include "../utils/pp_dimension_tree.h"
#include <ctf.hpp>
#include <fstream>

using namespace CTF;

template <typename dtype>
class CPPPOptimizer : virtual public CPDTOptimizer<dtype> {

public:
  CPPPOptimizer(int order, int r, World &dw, double tol_restart_dt, bool use_msdt);

  ~CPPPOptimizer();

  double step();

  double step_dt();

  double step_pp();

  void configure(Tensor<dtype> *input, Matrix<dtype> **mat, Matrix<dtype> *grad,
                 double lambda);

  void mttkrp_approx(int i, Matrix<> **dW, Matrix<> *N);

  void mttkrp_approx_second_correction(int i, Matrix<> &S, Matrix<> &S_temp,
                                       Matrix<> **WTW, Matrix<> **WTdW);

  double tol_restart_dt;
  bool restart;
  bool pp = false;
  bool reinitialize_tree;

  Matrix<> **dW;
  Matrix<> **update_W = NULL;
  Matrix<dtype> **WTdW = NULL;

  PPDimensionTree *ppdt = NULL;
};

#include "cp_pp_optimizer.cxx"

#endif

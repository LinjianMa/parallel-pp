#ifndef __CP_PP_LOCAL_OPTIMIZER_H__
#define __CP_PP_LOCAL_OPTIMIZER_H__

#include "../utils/dimension_tree.h"
#include <ctf.hpp>
#include <fstream>

using namespace CTF;

template <typename dtype>
class CPPPLocalOptimizer : public CPPPOptimizer<dtype>,
                           public CPDTLocalOptimizer<dtype> {

public:
  CPPPLocalOptimizer(int order, int r, World &dw, double tol_restart_dt);

  ~CPPPLocalOptimizer();

  double step();

  double step_dt();

  double step_pp();

  void configure(Tensor<dtype> *input, Matrix<dtype> **mat, Matrix<dtype> *grad,
                 double lambda);

  Matrix<> **dW_local;
};

#include "cp_pp_local_optimizer.cxx"

#endif

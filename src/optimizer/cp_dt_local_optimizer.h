#ifndef __CP_DT_LOCAL_OPTIMIZER_H__
#define __CP_DT_LOCAL_OPTIMIZER_H__

#include "../utils/dimension_tree.h"
#include "local_mttkrp.h"
#include <ctf.hpp>
#include <fstream>

using namespace CTF;

template <typename dtype>
class CPDTLocalOptimizer : virtual public CPDTOptimizer<dtype> {

public:
  CPDTLocalOptimizer(int order, int r, World &dw, bool use_msdt);

  CPDTLocalOptimizer(int order, int r, World &dw, bool use_msdt,
                     bool renew_ppoperator);

  ~CPDTLocalOptimizer();

  void configure(Tensor<dtype> *input, Matrix<dtype> **mat, Matrix<dtype> *grad,
                 double lambda);

  double step();

  double step_dt();

  double step_msdt();

  void solve_one_mode(int i);

  LocalMTTKRP<dtype> *local_mttkrp = NULL;
};

#include "cp_dt_local_optimizer.cxx"

#endif

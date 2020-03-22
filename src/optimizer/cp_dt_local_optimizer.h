#ifndef __CP_DT_LOCAL_OPTIMIZER_H__
#define __CP_DT_LOCAL_OPTIMIZER_H__

#include "../utils/dimension_tree.h"
#include <ctf.hpp>
#include <fstream>

using namespace CTF;

template <typename dtype>
class CPDTLocalOptimizer : public CPDTOptimizer<dtype> {

public:
  CPDTLocalOptimizer(int order, int r, World &dw);

  ~CPDTLocalOptimizer();

  void configure(Tensor<dtype> *input, Matrix<dtype> **mat, Matrix<dtype> *grad,
                 double lambda);

  double step();
};

#include "cp_dt_local_optimizer.cxx"

#endif

#ifndef __CP_SIMPLE_LOCAL_OPTIMIZER_H__
#define __CP_SIMPLE_LOCAL_OPTIMIZER_H__

#include "local_mttkrp.h"
#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class CPLocalOptimizer : public CPOptimizer<dtype> {

public:
  CPLocalOptimizer(int order, int r, World &dw);

  ~CPLocalOptimizer();

  void configure(Tensor<dtype> *input, Matrix<dtype> *mat, Matrix<dtype> *grad,
                 double lambda);

  double step();

  char seq_V[100];

  LocalMTTKRP<dtype> *local_mttkrp = NULL;
};

#include "cp_simple_local_optimizer.cxx"

#endif

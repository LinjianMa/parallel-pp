#ifndef __CP_SIMPLE_LOCAL_OPTIMIZER_H__
#define __CP_SIMPLE_LOCAL_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class CPLocalOptimizer : public CPOptimizer<dtype> {

public:
  CPLocalOptimizer(int order, int r, World &dw);

  ~CPLocalOptimizer();

  double step();

  char seq_V[100];
};

#include "cp_simple_local_optimizer.cxx"

#endif

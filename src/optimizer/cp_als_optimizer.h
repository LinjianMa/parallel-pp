#ifndef __CP_ALS_OPTIMIZER_H__
#define __CP_ALS_OPTIMIZER_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class CPOptimizer {

public:
  CPOptimizer(int order, int r, World &dw);

  ~CPOptimizer();

  void configure(Tensor<dtype> *input, Matrix<dtype> **mat, Matrix<dtype> *grad,
                 double lambda);

  void update_S(int update_index);

  void update_S_residual_calc();

  int order;
  int rank;
  // V: input tensor
  Tensor<dtype> *V = NULL;
  // W: output solutions
  Matrix<dtype> **W = NULL;
  Matrix<dtype> **M = NULL;
  // grad_W: gradient matrices
  Matrix<dtype> *grad_W = NULL;

  World *world;

  /*  initialize matrix S
   *   S["ij"] =
   * W[0]["ki"]*W[0]["kj"]*W[1]["ki"]*W[1]["kj"]*W[2]["ki"]*W[2]["kj"]*W[3]["ki"]*...
   */
  Matrix<dtype> S;
  Matrix<dtype> **WTW = NULL;
  Matrix<dtype> regul;
};

#include "cp_als_optimizer.cxx"

#endif

#ifndef __LOCAL_MTTKRP_H__
#define __LOCAL_MTTKRP_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class LocalMTTKRP {

public:
  LocalMTTKRP(int order, int r, World &dw);

  ~LocalMTTKRP();

  void distribute_mats(int mode);

  void post_mttkrp_reduce(int mode);

  void setup(Tensor<dtype> *T, Matrix<dtype> *mat_list);

  int order;
  int rank;
  // V: input tensor
  Tensor<dtype> *V = NULL;
  // W: output solutions
  Matrix<dtype> *W = NULL;

  // redistributed factored matrices
  Tensor<dtype> **redist_mats = NULL;

  World *world;

  // arrs[mode] is the local data for W.
  dtype **arrs = NULL;

  // physical index of each dimension
  int64_t *ldas = NULL;

  // number of processors in each dimension
  int *phys_phase = NULL;

  // number of pairs in V
  int64_t npair;
  Pair<dtype> *pairs = NULL;

  Partition par;

  char *par_idx = NULL;
};

#include "local_mttkrp.cxx"

#endif

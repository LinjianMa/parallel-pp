#ifndef __LOCAL_MTTKRP_H__
#define __LOCAL_MTTKRP_H__

#include <ctf.hpp>
#include <fstream>
using namespace CTF;

template <typename dtype> class LocalMTTKRP {

public:
  LocalMTTKRP(int order, int r, World &dw);

  ~LocalMTTKRP();

  void distribute_W(int i, Matrix<> **W, Matrix<> **W_local);

  void construct_W_remap(Matrix<> **W, Matrix<> **W_remap);

  void setup_V_local_data();

  void get_V_local_transposes();

  void construct_mttkrp_locals();

  void construct_mttkrp_reduce_communicators();

  void mttkrp_calc(int mode);

  void post_mttkrp_reduce(int mode);

  void setup(Tensor<dtype> *T, Matrix<dtype> **mat_list);

  int order;
  int rank;
  // V: input tensor
  Tensor<dtype> *V = NULL;
  Tensor<dtype> *V_local = NULL;

  // W: output solutions
  Matrix<dtype> **W = NULL;
  Matrix<dtype> **W_remap = NULL;
  Matrix<dtype> **W_local = NULL;
  // arrs[i] is the local data for W.
  dtype **arrs = NULL;

  Matrix<dtype> **mttkrp = NULL;
  Matrix<dtype> **mttkrp_local_mat = NULL;
  // arrs_mttkrp[i] is the local data for mttkrp.
  dtype **arrs_mttkrp = NULL;

  World *world;
  World *sworld;

  // physical index of each dimension
  int64_t *ldas = NULL;

  // number of processors in each dimension
  int *phys_phase = NULL;

  Partition par;

  char *par_idx = NULL;

  // The map storing the transposes of the input tensor (used in MSDT).
  map<int, Tensor<dtype> *> trans_V_local_map;
  map<int, string> trans_V_str_map;

  // Save the MPI communicators used in mttkrp_reduce
  MPI_Comm **cm_reduce = NULL;
  int **cmr_reduce = NULL;
};

#include "local_mttkrp.cxx"

#endif


#include "../utils/common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPPPOptimizer<dtype>::CPPPOptimizer(int order, int r, World &dw,
                                    double tol_restart_dt, bool use_msdt,
                                    bool renew_ppoperator, int ppmethod)
    : CPDTOptimizer<dtype>(order, r, dw, use_msdt, renew_ppoperator) {
  this->ppmethod = ppmethod;
  this->tol_restart_dt = tol_restart_dt;
  this->dW = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
  update_W = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
  WTdW = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
}

template <typename dtype> CPPPOptimizer<dtype>::~CPPPOptimizer() {
  // delete S;
  for (int i = 0; i < this->order; i++) {
    delete this->dW[i];
    delete this->update_W[i];
    delete this->WTdW[i];
  }
  free(this->dW);
  free(this->update_W);
  free(this->WTdW);
}

template <typename dtype>
void CPPPOptimizer<dtype>::configure(Tensor<dtype> *input, Matrix<dtype> **mat,
                                     Matrix<dtype> *grad, double lambda) {

  CPDTOptimizer<dtype>::configure(input, mat, grad, lambda);
  for (int i = 0; i < this->order; i++) {
    this->dW[i] = new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
    update_W[i] = new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
    WTdW[i] = new Matrix<>(this->rank, this->rank, *this->world);
  }

  ppdt = new PPDimensionTree(this->order, this->world, input, this->ppmethod);
}

template <typename dtype> double CPPPOptimizer<dtype>::step_dt() {
  Timer t_pp_step_dt("pp_step_dt");
  t_pp_step_dt.start();
  double num_sweep = 0.;

  if (this->world->rank == 0) {
    cout << "***** dt step *****" << endl;
  }

  for (int i = 0; i < this->order; i++) {
    this->dW[i]->operator[]("ij") = this->W[i]->operator[]("ij");
  }

  if (this->use_msdt == false) {
    CPDTOptimizer<dtype>::step();
    CPDTOptimizer<dtype>::step();
    num_sweep = 1.;
  } else {
    for (int i = 0; i < this->order; i++) {
      CPDTOptimizer<dtype>::step();
    }
    num_sweep = 1. * (this->order - 1);
  }

  int num_smallupdate = 0;
  for (int i = 0; i < this->order; i++) {
    this->dW[i]->operator[]("ij") -= this->W[i]->operator[]("ij");
    double dW_norm = this->dW[i]->norm2();
    double W_norm = this->W[i]->norm2();

    if (dW_norm / W_norm < this->tol_restart_dt) {
      num_smallupdate += 1;
    }
  }

  if (num_smallupdate == this->order) {
    this->pp = true;
    this->reinitialize_tree = true;
    if (this->renew_ppoperator == true) {
      // prepare inter_for_pp
      this->ppdt->inter_for_pp = this->inter_for_pp;
    }
  }

  t_pp_step_dt.stop();
  return num_sweep;
}

template <typename dtype>
void CPPPOptimizer<dtype>::mttkrp_approx(int i, Matrix<> **dW, Matrix<> *N) {
  Timer t_pp_mttkrp_approx("pp_approx_multi-TTV");
  t_pp_mttkrp_approx.start();

  vector<int> node_index = {i};
  string nodename = ppdt->get_nodename(node_index);
  N->operator[]("ij") = ppdt->name_tensor_map[nodename]->operator[]("ij");
  // construct parent vector. This sequence is to avoid cache miss.
  vector<vector<int>> parent_vec = {};
  int j1 = (i + this->order - 1) % this->order;
  int j2 = (i + 1) % this->order;
  vector<int> parent_index1;
  vector<int> parent_index2;
  for (auto const &parent_index : ppdt->pp_operator_indices) {
    if ((i == parent_index[0] && j1 == parent_index[1]) || (i == parent_index[1] && j1 == parent_index[0])) {
      parent_index1 = parent_index;
      break;
    }
  }
  for (auto const &parent_index : ppdt->pp_operator_indices) {
    if ((i == parent_index[0] && j2 == parent_index[1]) || (i == parent_index[1] && j2 == parent_index[0])) {
      parent_index2 = parent_index;
      break;
    }
  }
  parent_vec.push_back(parent_index1);
  for (auto const &parent_index : ppdt->pp_operator_indices) {
    if ((i == parent_index[0] || i == parent_index[1]) && parent_index != parent_index1 && parent_index != parent_index2) {
      parent_vec.push_back(parent_index);
    }
  }
  parent_vec.push_back(parent_index2);

  for (auto const &parent_index : parent_vec) {
      int j = parent_index[0];
      if (i == parent_index[0]) j = parent_index[1];
      string parentname = ppdt->get_nodename(parent_index);
      vector<string> einstr = ppdt->get_einstr(node_index, parent_index, j);
      char const *parent_str = einstr[0].c_str();
      char const *mat_str = einstr[1].c_str();
      char const *out_str = einstr[2].c_str();
      N->operator[](out_str) +=
          ppdt->name_tensor_map[parentname]->operator[](parent_str) *
          dW[j]->operator[](mat_str);
  }

  t_pp_mttkrp_approx.stop();
}

template <typename dtype>
void CPPPOptimizer<dtype>::mttkrp_approx_second_correction(int i, Matrix<> &S,
                                                           Matrix<> &S_temp,
                                                           Matrix<> **WTW,
                                                           Matrix<> **WTdW) {
  Timer t_pp_mttkrp_approx("pp_approx_hadamard_prod");
  t_pp_mttkrp_approx.start();

  vector<int> j_list = {};
  for (int j = 0; j < this->order; j++) {
    if (j != i) {
      j_list.push_back(j);
    }
  }
  S["ij"] = 0.;
  vector<vector<int>> dW_indices_list = subsets(j_list, 2);
  for (auto const &dW_indices : dW_indices_list) {
    S_temp["ij"] = 1.;
    for (auto const &j : j_list) {
      if (find(dW_indices.begin(), dW_indices.end(), j) != dW_indices.end()) {
        S_temp["ij"] *= WTdW[j]->operator[]("ij");
      } else {
        S_temp["ij"] *= WTW[j]->operator[]("ij");
      }
    }
    S["ij"] += S_temp["ij"];
  }
  if (this->ppmethod == 1) {
    // first order correction
    for (auto const &ii : j_list) {
      if (ii != (i+1)%this->order && ii != (i+this->order-1)%this->order) {
        S_temp["ij"] = 1.;
        for (auto const &j : j_list) {
          if (j == ii) {
            S_temp["ij"] *= WTdW[j]->operator[]("ij");
          } else {
            S_temp["ij"] *= (WTW[j]->operator[]("ij") - WTdW[j]->operator[]("ij"));
          }
        }
        S["ij"] += S_temp["ij"];
      }
    }
  }

  t_pp_mttkrp_approx.stop();
}

template <typename dtype> double CPPPOptimizer<dtype>::step_pp() {
  Timer t_pp_step_pp("pp_step_pp");
  t_pp_step_pp.start();

  if (this->world->rank == 0) {
    cout << "***** pairwise perturbation step *****" << endl;
  }

  for (int i = 0; i < this->order; i++) {
    mttkrp_approx(i, this->dW, this->M[i]);
    Matrix<> S_temp = Matrix<>(this->rank, this->rank, *this->world);
    mttkrp_approx_second_correction(i, this->S, S_temp, this->WTW, this->WTdW);
    this->M[i]->operator[]("ij") +=
        this->W[i]->operator[]("ik") * this->S["kj"];

    CPOptimizer<dtype>::update_S(i);
    spd_solve(*this->M[i], *this->update_W[i], this->S);
    this->WTW[i]->operator[]("jk") = this->update_W[i]->operator[]("ij") *
                                     this->update_W[i]->operator[]("ik");

    this->dW[i]->operator[]("ij") +=
        update_W[i]->operator[]("ij") - this->W[i]->operator[]("ij");
    this->W[i]->operator[]("ij") = update_W[i]->operator[]("ij");

    this->WTdW[i]->operator[]("jk") =
        this->W[i]->operator[]("ij") * this->dW[i]->operator[]("ik");
  }

  int num_bigupdate = 0;
  for (int i = 0; i < this->order; i++) {
    double dW_norm = this->dW[i]->norm2();
    double W_norm = this->W[i]->norm2();

    if (dW_norm / W_norm > this->tol_restart_dt) {
      num_bigupdate += 1;
    }
  }
  if (num_bigupdate > 0) {
    this->pp = false;
    this->reinitialize_tree = false;
  }

  t_pp_step_pp.stop();
  return 1.;
}

template <typename dtype> double CPPPOptimizer<dtype>::step() {

  double num_sweep = 0.;

  this->restart = false;

  if (this->pp == true) {
    if (this->reinitialize_tree == true) {
      this->restart = true;
      for (int i = 0; i < this->order; i++) {
        this->dW[i]->operator[]("ij") = 0.;
        this->WTdW[i]->operator[]("ij") = 0.;
      }
      ppdt->initialize_tree(this->W);
      this->reinitialize_tree = false;
    }
    num_sweep = step_pp();
  } else {
    num_sweep = step_dt();
  }

  return num_sweep;
}

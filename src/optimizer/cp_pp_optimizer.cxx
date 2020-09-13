
#include "../utils/common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
CPPPOptimizer<dtype>::CPPPOptimizer(int order, int r, World &dw,
                                    double tol_restart_dt)
    : CPDTOptimizer<dtype>(order, r, dw, false) {
  this->tol_restart_dt = tol_restart_dt;
  this->dW = (Matrix<> **)malloc(order * sizeof(Matrix<> *));
}

template <typename dtype> CPPPOptimizer<dtype>::~CPPPOptimizer() {
  // delete S;
  free(this->dW);
}

template <typename dtype>
void CPPPOptimizer<dtype>::configure(Tensor<dtype> *input, Matrix<dtype> **mat,
                                     Matrix<dtype> *grad, double lambda) {

  CPDTOptimizer<dtype>::configure(input, mat, grad, lambda);
  for (int i = 0; i < this->order; i++) {
    this->dW[i] = new Matrix<>(this->W[i]->nrow, this->rank, *this->world);
  }
}

template <typename dtype> double CPPPOptimizer<dtype>::step_dt() {
  Timer t_pp_step_dt("pp_step_dt");
  t_pp_step_dt.start();

  if (this->world->rank == 0) {
    cout << "***** dt step *****" << endl;
  }

  for (int i = 0; i < this->order; i++) {
    this->dW[i]->operator[]("ij") = this->W[i]->operator[]("ij");
  }

  CPDTOptimizer<dtype>::step();
  CPDTOptimizer<dtype>::step();

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
  }

  t_pp_step_dt.stop();
  return 1.;
}

template <typename dtype>
Matrix<> CPPPOptimizer<dtype>::mttkrp_approx(int i, Matrix<> **dW) {
  Timer t_pp_mttkrp_approx("pp_mttkrp_approx");
  t_pp_mttkrp_approx.start();

  vector<int> node_index = {i};
  string nodename = get_nodename(node_index);

  Matrix<> N = Matrix<>(*this->name_tensor_map[nodename]);
  for (int j = 0; j < i; j++) {
    vector<int> parent_index = {j, i};
    string parentname = get_nodename(parent_index);
    vector<string> einstr = get_einstr(node_index, parent_index, j);
    char const *parent_str = einstr[0].c_str();
    char const *mat_str = einstr[1].c_str();
    char const *out_str = einstr[2].c_str();
    N[out_str] += this->name_tensor_map[parentname]->operator[](parent_str) *
                         dW[j]->operator[](mat_str);
  }
  for (int j = i + 1; j < this->order; j++) {
    vector<int> parent_index = {i, j};
    string parentname = get_nodename(parent_index);
    vector<string> einstr = get_einstr(node_index, parent_index, j);
    char const *parent_str = einstr[0].c_str();
    char const *mat_str = einstr[1].c_str();
    char const *out_str = einstr[2].c_str();
    N[out_str] += this->name_tensor_map[parentname]->operator[](parent_str) *
                         dW[j]->operator[](mat_str);
  }

  t_pp_mttkrp_approx.stop();
  return N;
}

template <typename dtype> double CPPPOptimizer<dtype>::step_pp() {
  Timer t_pp_step_pp("pp_step_pp");
  t_pp_step_pp.start();

  if (this->world->rank == 0) {
    cout << "***** pairwise perturbation step *****" << endl;
  }

  for (int i = 0; i < this->order; i++) {
    Matrix<> mttkrp_temp = mttkrp_approx(i, this->dW);
    this->M[i]->operator[]("ij") = mttkrp_temp["ij"];
    CPOptimizer<dtype>::update_S(i);
    Matrix<> update_W = Matrix<>(*this->W[i]);
    spd_solve(*this->M[i], update_W, this->S);
    this->dW[i]->operator[]("ij") += update_W["ij"] -
                                    this->W[i]->operator[]("ij");
    this->W[i]->operator[]("ij") = update_W["ij"];
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

template <typename dtype>
string CPPPOptimizer<dtype>::get_nodename(vector<int> nodeindex) {
  /*
    Example:
      When the input tensor has 4 dimensions:
      get_nodename([1,2]) == 'bc'
  */
  char name[100];
  if (nodeindex.size() == this->order) {
    name[0] = '0';
    name[1] = '\0';
  } else {
    for (int i = 0; i < nodeindex.size(); i++) {
      name[i] = 'a' + nodeindex[i];
    }
    name[nodeindex.size()] = '\0';
  }
  return string(name);
}

template <typename dtype>
vector<string> CPPPOptimizer<dtype>::get_einstr(vector<int> nodeindex,
                                                vector<int> parent_nodeindex,
                                                int contract_index) {
  /*
    Example:
      When the input tensor has 4 dimensions:
      _get_einstr([1,2], [1,2,3], 3) == ["abcR", "cR", "abR"]
  */
  string ci = "";
  if (parent_nodeindex.size() != this->order) {
    ci = "R";
  }

  char str1[100], str2[3], str3[100];

  for (int i = 0; i < parent_nodeindex.size(); i++) {
    str1[i] = 'a' + parent_nodeindex[i];
  }
  if (ci == "R") {
    str1[parent_nodeindex.size()] = 'R';
    str1[parent_nodeindex.size() + 1] = '\0';
  } else {
    str1[parent_nodeindex.size()] = '\0';
  }

  str2[0] = 'a' + contract_index;
  str2[1] = 'R';
  str2[2] = '\0';

  for (int i = 0; i < nodeindex.size(); i++) {
    str3[i] = 'a' + nodeindex[i];
  }
  str3[nodeindex.size()] = 'R';
  str3[nodeindex.size() + 1] = '\0';

  vector<string> ret{string(str1), string(str2), string(str3)};
  return ret;
}

template <typename dtype>
void CPPPOptimizer<dtype>::get_parentnode(vector<int> nodeindex,
                                          string &parent_nodename,
                                          vector<int> &parent_index,
                                          int &contract_index) {

  vector<int> fulllist = {};
  for (int i = 0; i < this->order; i++) {
    fulllist.push_back(i);
  }

  vector<int> comp_index(this->order);
  vector<int>::iterator it;

  // comp_index = np.setdiff1d(fulllist, nodeindex)
  it = set_difference(fulllist.begin(), fulllist.end(), nodeindex.begin(),
                      nodeindex.end(), comp_index.begin());
  comp_index.resize(it - comp_index.begin());
  vector<int> comp_parent_index(comp_index.begin() + 1, comp_index.end());
  contract_index = comp_index[0];

  // parent_index = np.setdiff1d(fulllist, comp_parent_index)
  it = set_difference(fulllist.begin(), fulllist.end(),
                      comp_parent_index.begin(), comp_parent_index.end(),
                      parent_index.begin());
  parent_index.resize(it - parent_index.begin());

  parent_nodename = get_nodename(parent_index);
}

template <typename dtype>
void CPPPOptimizer<dtype>::initialize_treenode(vector<int> nodeindex, World *dw,
                                               Tensor<> *T, Matrix<> **mat) {
  Timer t_pp_initialize_treenode("pp_initialize_treenode");
  t_pp_initialize_treenode.start();

  string nodename = get_nodename(nodeindex);

  string parent_nodename;
  vector<int> parent_nodeindex(this->order);
  int contract_index;
  get_parentnode(nodeindex, parent_nodename, parent_nodeindex, contract_index);

  vector<string> einstr =
      get_einstr(nodeindex, parent_nodeindex, contract_index);
  char const *parent_str = einstr[0].c_str();
  char const *mat_str = einstr[1].c_str();
  char const *out_str = einstr[2].c_str();

  if (name_index_map.find(parent_nodename) == name_index_map.end()) {
    initialize_treenode(parent_nodeindex, dw, T, mat);
  }

  // store that into the name_tensor_map
  int lens[strlen(out_str)];
  for (int ii = 0; ii < strlen(out_str); ii++) {
    if (out_str[ii] == 'R')
      lens[ii] = mat[0]->ncol;
    else
      lens[ii] = T->lens[int(out_str[ii] - 'a')];
  }
  name_tensor_map[nodename] = new Tensor<dtype>(strlen(out_str), lens, *dw);
  name_index_map[nodename] = nodeindex;

  name_tensor_map[nodename]->operator[](out_str) +=
      name_tensor_map[parent_nodename]->operator[](parent_str) *
      mat[contract_index]->operator[](mat_str);

  t_pp_initialize_treenode.stop();
}

template <typename dtype>
void CPPPOptimizer<dtype>::initialize_tree(World *dw, Tensor<> *T,
                                           Matrix<> **mat, Matrix<> **deltaW) {
  Timer t_pp_initialize_tree("pp_initialize_tree");
  t_pp_initialize_tree.start();

  for (auto const &x : this->name_tensor_map) {
    // Note that "0" is the input tensor, and we cannot delete it.
    if (x.first != "0") {
      delete x.second;
    }
  }
  name_tensor_map.clear();
  name_index_map.clear();
  vector<int> fulllist = {};
  for (int i = 0; i < this->order; i++) {
    fulllist.push_back(i);
  }
  name_index_map["0"] = fulllist;
  name_tensor_map["0"] = T;

  for (int i = 0; i < this->order; i++) {
    deltaW[i]->operator[]("ij") = 0.;
  }
  for (int ii = 0; ii < this->order; ii++)
    for (int jj = ii + 1; jj < this->order; jj++) {
      vector<int> nodeindex = {ii, jj};
      initialize_treenode(nodeindex, dw, T, mat);
    }
  for (int ii = 0; ii < this->order; ii++) {
    vector<int> nodeindex = {ii};
    initialize_treenode(nodeindex, dw, T, mat);
  }

  t_pp_initialize_tree.stop();
}

template <typename dtype> double CPPPOptimizer<dtype>::step() {

  double num_sweep = 0.;

  this->restart = false;

  if (this->pp == true) {
    if (this->reinitialize_tree == true) {
      this->restart = true;
      initialize_tree(this->world, this->V, this->W, this->dW);
      this->reinitialize_tree = false;
    }
    num_sweep = step_pp();
  } else {
    num_sweep = step_dt();
  }

  return num_sweep;
}

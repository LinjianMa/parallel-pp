
#include "pp_dimension_tree.h"
#include "common.h"
#include <ctf.hpp>

using namespace CTF;

PPDimensionTree::PPDimensionTree(int order, World *world, Tensor<> *T) {
  this->order = order;
  this->world = world;
  this->T = T;
}

PPDimensionTree::PPDimensionTree(int order, World *world, Tensor<> *T,
                                 map<int, Tensor<> *> trans_T_map,
                                 map<int, string> trans_T_str_map) {
  this->order = order;
  this->world = world;
  this->T = T;
  this->trans_T_map = trans_T_map;
  this->trans_T_str_map = trans_T_str_map;
}

PPDimensionTree::~PPDimensionTree() {}

string PPDimensionTree::get_nodename(vector<int> nodeindex) {
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

vector<string> PPDimensionTree::get_einstr(vector<int> nodeindex,
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

void PPDimensionTree::get_parentnode(vector<int> nodeindex,
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

void PPDimensionTree::initialize_treenode(vector<int> nodeindex, World *dw,
                                          Tensor<> *T, Matrix<> **mat) {
  Timer t_pp_initialize_treenode("pp_initialize_treenode");
  t_pp_initialize_treenode.start();

  string nodename = get_nodename(nodeindex);

  if (this->world->rank == 0) {
    cout << "pp nodename is: " << nodename << endl;
  }

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

  if (name_tensor_map.find(nodename) == name_tensor_map.end()) {
    // store that into the name_tensor_map
    int lens[strlen(out_str)];
    for (int ii = 0; ii < strlen(out_str); ii++) {
      if (out_str[ii] == 'R')
        lens[ii] = mat[0]->ncol;
      else
        lens[ii] = T->lens[int(out_str[ii] - 'a')];
    }
    name_tensor_map[nodename] = new Tensor<>(strlen(out_str), lens, *dw);
    name_tensor_map[nodename]->operator[](out_str) +=
        name_tensor_map[parent_nodename]->operator[](parent_str) *
        mat[contract_index]->operator[](mat_str);
  } else {
    name_tensor_map[nodename]->operator[](out_str) =
        name_tensor_map[parent_nodename]->operator[](parent_str) *
        mat[contract_index]->operator[](mat_str);
  }
  name_index_map[nodename] = nodeindex;

  t_pp_initialize_treenode.stop();
}

void PPDimensionTree::initialize_tree(World *dw, Tensor<> *T, Matrix<> **mat) {
  Timer t_pp_initialize_tree("pp_initialize_tree");
  t_pp_initialize_tree.start();

  name_index_map.clear();

  vector<int> fulllist = {};
  for (int i = 0; i < this->order; i++) {
    fulllist.push_back(i);
  }
  name_index_map["0"] = fulllist;
  name_tensor_map["0"] = T;

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
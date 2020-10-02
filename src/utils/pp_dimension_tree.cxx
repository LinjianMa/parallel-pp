
#include "pp_dimension_tree.h"
#include "common.h"
#include <ctf.hpp>

using namespace CTF;

PPDimensionTree::PPDimensionTree(int order, World *world, Tensor<> *T) {
  this->order = order;
  this->world = world;
  this->T = T;

  for (int i = 0; i < this->order; i++) {
    this->fulllist.push_back(i);
  }
  dw = World();
  construct_pp_operator_indices();
}

PPDimensionTree::PPDimensionTree(int order, World *world, Tensor<> *T,
                                 map<int, Tensor<> *> trans_T_map,
                                 map<int, string> trans_T_str_map)
    : PPDimensionTree(order, world, T) {
  this->trans_T_map = trans_T_map;
  this->trans_T_str_map = trans_T_str_map;
  if (trans_T_str_map.size() != 0) {
    this->use_transpose_T = true;
  }
}

PPDimensionTree::~PPDimensionTree() {}

void PPDimensionTree::construct_pp_operator_indices() {
  for (int ii = 0; ii < this->order; ii++)
    for (int jj = ii + 1; jj < this->order; jj++) {
      vector<int> nodeindex = {jj, ii};
      if (jj - ii < ii + this->order - jj) {
        nodeindex = {ii, jj};
      }
      this->pp_operator_indices.push_back(nodeindex);
    }
}

string PPDimensionTree::get_nodename(vector<int> nodeindex) {
  /*
    Example:
      When the input tensor has 4 dimensions:
      get_nodename([1,2]) == 'bc'
  */
  char name[100];
  for (int i = 0; i < nodeindex.size(); i++) {
    name[i] = 'a' + nodeindex[i];
  }
  name[nodeindex.size()] = '\0';
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
  // matrix -> pp operator
  if (nodeindex.size() == 1) {
    for (auto const &index : this->pp_operator_indices) {
      if (nodeindex[0] == index[0] || nodeindex[0] == index[1]) {
        parent_index = index;
        parent_nodename = get_nodename(parent_index);
        contract_index = index[0];
        if (nodeindex[0] == index[0]) {
          contract_index = index[1];
        }
        return;
      }
    }
  }
  // second level intermediate -> first level intermediate
  if (nodeindex.size() == this->order - 2) {
    vector<int> sort_nodeindex = nodeindex;
    sort(sort_nodeindex.begin(), sort_nodeindex.end());
    vector<int>::iterator it;
    for (auto const &x : this->name_index_map) {
      if (x.second.size() == this->order - 1) {
        vector<int> sort_parentlist = x.second;
        sort(sort_parentlist.begin(), sort_parentlist.end());
        vector<int> comp_index(this->order);
        it = set_difference(sort_parentlist.begin(), sort_parentlist.end(), sort_nodeindex.begin(), sort_nodeindex.end(), comp_index.begin());
        comp_index.resize(it - comp_index.begin());
        if (comp_index.size() == 1) {
          contract_index = comp_index[0];
          parent_index = x.second;
          parent_nodename = get_nodename(parent_index);
          return;
        }
      }
    }
  }

  vector<int> comp_index(this->order);
  vector<int>::iterator it;

  vector<int> sort_nodeindex = nodeindex;
  sort(sort_nodeindex.begin(), sort_nodeindex.end());

  // comp_index = np.setdiff1d(fulllist, nodeindex)
  it = set_difference(fulllist.begin(), fulllist.end(), sort_nodeindex.begin(),
                      sort_nodeindex.end(), comp_index.begin());
  comp_index.resize(it - comp_index.begin());

  if (comp_index.size() == 1 && this->use_transpose_T == true) {
    contract_index = comp_index[0];
    parent_nodename = this->trans_T_str_map[contract_index];
    parent_index = this->name_index_map[parent_nodename];
  } else {
    vector<int> comp_parent_index(comp_index.begin(), comp_index.end() - 1);
    contract_index = comp_index[comp_index.size() - 1];
    // parent_index = np.setdiff1d(fulllist, comp_parent_index)
    it = set_difference(fulllist.begin(), fulllist.end(),
                        comp_parent_index.begin(), comp_parent_index.end(),
                        parent_index.begin());
    parent_index.resize(it - parent_index.begin());
    parent_nodename = get_nodename(parent_index);
  }
}

void PPDimensionTree::initialize_treenode(vector<int> nodeindex,
                                          Matrix<> **mat) {
  string nodename = get_nodename(nodeindex);
  if (name_index_map.find(nodename) != name_index_map.end()) {
    return;
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
    initialize_treenode(parent_nodeindex, mat);
  }

  if (name_tensor_map.find(nodename) == name_tensor_map.end()) {
    // store that into the name_tensor_map
    int lens[strlen(out_str)];
    for (int ii = 0; ii < strlen(out_str); ii++) {
      if (out_str[ii] == 'R')
        lens[ii] = mat[0]->ncol;
      else
        lens[ii] = this->T->lens[int(out_str[ii] - 'a')];
    }
    name_tensor_map[nodename] =
        new Tensor<>(strlen(out_str), lens, *this->world);
    name_tensor_map[nodename]->operator[](out_str) +=
        name_tensor_map[parent_nodename]->operator[](parent_str) *
        mat[contract_index]->operator[](mat_str);
  } else {
    name_tensor_map[nodename]->operator[](out_str) =
        name_tensor_map[parent_nodename]->operator[](parent_str) *
        mat[contract_index]->operator[](mat_str);
  }
  name_index_map[nodename] = nodeindex;

  if (dw.rank == 0) {
    cout << "nodename is: " << nodename
         << "   parent nodename is: " << parent_nodename << endl;
    cout << "einstr is: " << out_str << "=" << parent_str << "," << mat_str
         << endl;
  }
}

void PPDimensionTree::initialize_tree_root(Matrix<> **mat) {
  if (this->use_transpose_T == false) {
    string nodename = get_nodename(this->fulllist);
    this->name_index_map[nodename] = this->fulllist;
    this->name_tensor_map[nodename] = this->T;
  } else {
    for (auto const &x : this->trans_T_str_map) {
      string nodename = x.second;
      vector<int> translist = {};
      for (int i = 0; i < this->order; i++) {
        translist.push_back(int(nodename[i] - 'a'));
      }

      this->name_index_map[nodename] = translist;
      this->name_tensor_map[nodename] = this->trans_T_map[x.first];
    }
  }
  // first level intermediates
  Timer pp_init_partial_MTTKRP("pp_init_partial-MTTKRP");
  for (int mode = 0; mode < 3; mode++) {
    vector<int> nodeindex;
    get_first_level_intermediate_parameters(mode, nodeindex);
    initialize_treenode(nodeindex, mat);
  }
  pp_init_partial_MTTKRP.stop();
}

void PPDimensionTree::get_first_level_intermediate_parameters(int mode, vector<int>& nodeindex) {
  for (int i = mode + 1; i < order; i++) {
    nodeindex.push_back(i);
  }
  for (int i = 0; i < mode; i++) {
    nodeindex.push_back(i);
  }
}

void PPDimensionTree::save_top_intermediate() {
  Timer t_pp_save_top_intermediate("pp_save_top_intermediate");
  t_pp_save_top_intermediate.start();
  for (int mode = 0; mode < this->inter_for_pp.size(); mode++) {
    vector<int> nodeindex;
    get_first_level_intermediate_parameters(mode, nodeindex);
    string nodename = get_nodename(nodeindex);
    int index = this->inter_for_pp.size() - 1 - mode;
    this->name_tensor_map[nodename] = this->inter_for_pp[index];
    this->name_index_map[nodename] = nodeindex;
    if (dw.rank == 0) {
      cout << "[save_top_intermediate] nodename is: " << nodename << endl;
    }
  }
  t_pp_save_top_intermediate.stop();
}

void PPDimensionTree::initialize_tree(Matrix<> **mat) {
  Timer t_pp_initialize_tree("pp_initialize_tree");
  t_pp_initialize_tree.start();

  if (dw.rank == 0) {
    cout << "***** pairwise perturbation initialize tree *****" << endl;
  }

  this->name_index_map.clear();
  if (this->inter_for_pp.size() != 0) {
    save_top_intermediate();
  }
  initialize_tree_root(mat);

  Timer pp_init_multi_TTV("pp_init_multi-TTV");
  for (auto const &nodeindex : this->pp_operator_indices) {
    initialize_treenode(nodeindex, mat);
  }
  for (int ii = 0; ii < this->order; ii++) {
    vector<int> nodeindex = {ii};
    initialize_treenode(nodeindex, mat);
  }
  pp_init_multi_TTV.stop();
  t_pp_initialize_tree.stop();
}
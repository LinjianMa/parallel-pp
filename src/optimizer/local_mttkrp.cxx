
#include "../utils/common.h"
#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
LocalMTTKRP<dtype>::LocalMTTKRP(int order, int r, World &dw) {
  this->world = &dw;
  this->sworld = new World(MPI_COMM_SELF);
  this->order = order;
  this->rank = r;
}

template <typename dtype> LocalMTTKRP<dtype>::~LocalMTTKRP() {
  // free(redist_mats);
  free(par_idx);
  free(phys_phase);
  free(ldas);
  free(arrs);
  free(arrs_mttkrp);
  free(mttkrp);
  delete V_local;
  for (int i = 0; i < this->order; i++) {
    delete W_local[i];
  }
  free(W_local);
  for (int i = 0; i < this->order; i++) {
    if (this->cm_reduce[i] != NULL) {
      MPI_Comm_free(this->cm_reduce[i]);
    }
  }
  free(this->cm_reduce);
  free(this->cmr_reduce);
}

template <typename dtype> void LocalMTTKRP<dtype>::setup_V_local_data() {
  int64_t pad_local_col[this->V->order];
  for (int j = 0; j < this->order; j++) {
    pad_local_col[j] = int(this->V->pad_edge_len[j] / this->phys_phase[j]);
  }
  this->V_local = new Tensor<>(this->V->order, pad_local_col, *this->sworld);
  int num_elements = 1;
  for (int j = 0; j < this->order; j++) {
    num_elements *= pad_local_col[j];
  }
  // TODO: currently it uses 2X memory.
  memcpy(this->V_local->data, this->V->data, sizeof(dtype) * num_elements);
}

template <typename dtype> void LocalMTTKRP<dtype>::get_V_local_transposes() {
  char seq_V[order + 1];
  seq_V[order] = '\0';
  for (int j = 0; j < order; j++) {
    seq_V[j] = 'a' + j;
  }

  int num_transposes = 0;
  int odd_mode = order % 2;
  if (odd_mode == 1) {
    num_transposes = int((order - 1) / 2);
  } else {
    num_transposes = int(order / 2);
  }

  trans_V_local_map[0] = this->V_local;
  trans_V_local_map[order - 1] = this->V_local;
  trans_V_str_map[0] = string(seq_V);
  trans_V_str_map[order - 1] = string(seq_V);

  for (int partial_trans = 1; partial_trans < num_transposes; partial_trans++) {
    char seq_trans[order + 1];
    int lens_out[order];
    seq_trans[order] = '\0';
    int j = 0;
    for (int i = 2 * partial_trans; i < order; i++) {
      seq_trans[j] = 'a' + i;
      lens_out[j] = this->V_local->lens[i];
      j++;
    }
    for (int i = 0; i < 2 * partial_trans; i++) {
      seq_trans[j] = 'a' + i;
      lens_out[j] = this->V_local->lens[i];
      j++;
    }
    Tensor<> *out = new Tensor<>(order, lens_out, *this->sworld);
    out->operator[](seq_trans) = this->V_local->operator[](seq_V);

    trans_V_local_map[2 * partial_trans] = out;
    trans_V_local_map[2 * partial_trans - 1] = out;
    trans_V_str_map[2 * partial_trans] = string(seq_trans);
    trans_V_str_map[2 * partial_trans - 1] = string(seq_trans);
  }
  if (odd_mode == 1) {
    char seq_trans[order + 1];
    int lens_out[order];
    seq_trans[order] = '\0';
    int j = 0;
    for (int i = order - 2; i < order; i++) {
      seq_trans[j] = 'a' + i;
      lens_out[j] = this->V_local->lens[i];
      j++;
    }
    for (int i = 0; i < order - 2; i++) {
      seq_trans[j] = 'a' + i;
      lens_out[j] = this->V_local->lens[i];
      j++;
    }
    Tensor<> *out = new Tensor<>(order, lens_out, *this->sworld);
    out->operator[](seq_trans) = this->V_local->operator[](seq_V);

    trans_V_local_map[order - 2] = out;
    trans_V_str_map[order - 2] = string(seq_trans);
  }
}

template <typename dtype> void LocalMTTKRP<dtype>::mttkrp_calc(int mode) {
  char seq_V[100];
  // make the char seq_V
  seq_V[this->order] = '\0';
  for (int j = 0; j < this->order; j++) {
    seq_V[j] = 'a' + j;
  }

  swap_char(seq_V, mode, order - 1);

  int index[this->order];
  int lens_H[this->order];
  for (int j = 0; j <= this->order - 1; j++) {
    index[j] = (int)(seq_V[j] - 'a');
    lens_H[j] = this->W_local[index[j]]->nrow;
    IASSERT(this->W_local[index[j]]->nrow == this->V_local->lens[index[j]]);
  }

  KhatriRao_contract(*(this->mttkrp_local_mat[mode]), *(this->V_local),
                     this->W_local, index, lens_H, *this->sworld);
}

template <typename dtype>
void LocalMTTKRP<dtype>::distribute_W(int i, Matrix<> **W, Matrix<> **W_local) {
  Timer t_mttkrp_remap("MTTKRP_distribute_W");
  t_mttkrp_remap.start();

  if (this->phys_phase[i] == 1) {
    // one process in dim i
    if (this->world->np == 1) {
      // overall only 1 process
      this->arrs[i] = (dtype *)W[i]->data;
    } else {
      // overall >1 processes
      this->W_remap[i]->operator[]("ij") = W[i]->operator[]("ij");
      arrs[i] = (dtype *)this->V->sr->alloc(this->V->lens[i] * this->rank);
      this->W_remap[i]->read_all(arrs[i], true);
    }
  } else {
    // multiple processes in dim i
    int topo_dim = V->edge_map[i].cdt;
    IASSERT(V->edge_map[i].type == CTF_int::PHYSICAL_MAP);
    IASSERT(!V->edge_map[i].has_child ||
            V->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);

    int comm_lda = 1;
    for (int l = 0; l < topo_dim; l++) {
      comm_lda *= V->topo->dim_comm[l].np;
    }
    CTF_int::CommData cmdt(V->wrld->rank -
                               comm_lda * V->topo->dim_comm[topo_dim].rank,
                           V->topo->dim_comm[topo_dim].rank, V->wrld->cdt);

    this->W_remap[i]->operator[]("ij") = W[i]->operator[]("ij");
    arrs[i] = (dtype *)this->W_remap[i]->data;
    cmdt.bcast(this->W_remap[i]->data, this->W_remap[i]->size, V->sr->mdtype(),
               0);
  }
  // update the W_local
  IASSERT(this->V->pad_edge_len[i] == this->W_remap[i]->pad_edge_len[0]);
  int64_t pad_local_col = int(this->V->pad_edge_len[i] / this->phys_phase[i]);
  memcpy(W_local[i]->data, (char *)arrs[i],
         sizeof(dtype) * pad_local_col * this->rank);

  t_mttkrp_remap.stop();
}

template <typename dtype>
void LocalMTTKRP<dtype>::construct_W_remap(Matrix<> **W, Matrix<> **W_remap) {
  for (int i = 0; i < this->order; i++) {
    if (this->phys_phase[i] == 1) {
      // one process in dim i
      if (this->world->np == 1) {
        // overall only 1 process
        W_remap[i] = new Matrix<dtype>(W[i]->nrow, this->rank, *this->world);
      } else {
        // overall >1 processes
        char nonastr[2];
        nonastr[0] = 'a' - 1;
        nonastr[1] = 'a' - 2;
        W_remap[i] =
            new Matrix<dtype>(W[i]->nrow, this->rank, nonastr, par[par_idx],
                              Idx_Partition(), 0, *this->world, *V->sr);
      }
    } else {
      // multiple processes in dim i
      int topo_dim = V->edge_map[i].cdt;
      IASSERT(V->edge_map[i].type == CTF_int::PHYSICAL_MAP);
      IASSERT(!V->edge_map[i].has_child ||
              V->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);

      char mat_idx[2];
      mat_idx[0] = par_idx[topo_dim];
      mat_idx[1] = 'a';

      W_remap[i] =
          new Matrix<dtype>(W[i]->nrow, this->rank, mat_idx, par[par_idx],
                            Idx_Partition(), 0, *V->wrld, *V->sr);
    }
  }
}

template <typename dtype> void LocalMTTKRP<dtype>::construct_mttkrp_locals() {

  Timer t_mttkrp_construction("mttkrp_construction");
  t_mttkrp_construction.start();

  for (int i = 0; i < order; i++) {

    if (this->phys_phase[i] == 1) {
      // one process in dim i
      if (this->world->np == 1) {
        // overall only 1 process
        this->mttkrp[i] =
            new Matrix<dtype>(this->W[i]->nrow, this->rank, *this->world);
        this->arrs_mttkrp[i] = (dtype *)this->mttkrp[i]->data;
      } else {
        // overall >1 processes
        char nonastr[2];
        nonastr[0] = 'a' - 1;
        nonastr[1] = 'a' - 2;
        this->mttkrp[i] = new Matrix<dtype>(
            this->W[i]->nrow, this->rank, nonastr, par[par_idx],
            Idx_Partition(), 0, *this->world, *V->sr);
        this->arrs_mttkrp[i] = (dtype *)this->mttkrp[i]->data;
      }
    } else {
      // multiple processes in dim i
      int topo_dim = V->edge_map[i].cdt;
      IASSERT(V->edge_map[i].type == CTF_int::PHYSICAL_MAP);
      IASSERT(!V->edge_map[i].has_child ||
              V->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);

      char mat_idx[2];
      mat_idx[0] = par_idx[topo_dim];
      mat_idx[1] = 'a';
      Matrix<dtype> *m =
          new Matrix<dtype>(this->W[i]->nrow, this->rank, mat_idx, par[par_idx],
                            Idx_Partition(), 0, *V->wrld, *V->sr);

      this->mttkrp[i] = m;
      this->arrs_mttkrp[i] = (dtype *)m->data;
    }
    // build the mttkrp_local_mat
    IASSERT(this->V->pad_edge_len[i] == this->W_remap[i]->pad_edge_len[0]);
    int64_t pad_local_col = int(this->V->pad_edge_len[i] / this->phys_phase[i]);
    this->mttkrp_local_mat[i] =
        new Matrix<dtype>(pad_local_col, this->rank, *sworld);
    char *tempdata = this->mttkrp_local_mat[i]->data;
    this->mttkrp_local_mat[i]->data = (char *)arrs_mttkrp[i];
    free(tempdata);
  }
  // construct communicators
  construct_mttkrp_reduce_communicators();
  t_mttkrp_construction.stop();
}

template <typename dtype>
void LocalMTTKRP<dtype>::post_mttkrp_reduce(int mode) {
  int red_len = this->world->np / this->phys_phase[mode];
  if (red_len > 1) {
    Timer t_mttkrp_red("MTTKRP_Reduce");
    t_mttkrp_red.start();
    int64_t sz = this->mttkrp[mode]->size;
    if (*this->cmr_reduce[mode] == 0) {
      MPI_Reduce(MPI_IN_PLACE, this->arrs_mttkrp[mode], sz,
                 this->V->sr->mdtype(), this->V->sr->addmop(), 0,
                 *this->cm_reduce[mode]);
    } else {
      MPI_Reduce(this->arrs_mttkrp[mode], NULL, sz, this->V->sr->mdtype(),
                 this->V->sr->addmop(), 0, *this->cm_reduce[mode]);
      std::fill(this->arrs_mttkrp[mode], this->arrs_mttkrp[mode] + sz,
                *((dtype *)this->V->sr->addid()));
    }
    t_mttkrp_red.stop();
  }
}

template <typename dtype>
void LocalMTTKRP<dtype>::construct_mttkrp_reduce_communicators() {
  Timer t_mttkrp_red_prep("MTTKRP_Reduce_prep");
  t_mttkrp_red_prep.start();

  for (int mode = 0; mode < this->V->order; mode++) {
    int red_len = this->world->np / this->phys_phase[mode];
    if (red_len > 1) {
      int jr = this->V->edge_map[mode].calc_phys_rank(this->V->topo);
      this->cm_reduce[mode] = new MPI_Comm();
      MPI_Comm_split(this->world->comm, jr, this->world->rank,
                     this->cm_reduce[mode]);
      this->cmr_reduce[mode] = new int();
      MPI_Comm_rank(*this->cm_reduce[mode], this->cmr_reduce[mode]);
    }
  }
  t_mttkrp_red_prep.stop();
}

template <typename dtype>
void LocalMTTKRP<dtype>::setup(Tensor<dtype> *V, Matrix<dtype> **mat_list) {

  IASSERT(mat_list[0]->ncol == this->rank);
  IASSERT(this->order == V->order);
  for (int i = 0; i < this->order; i++) {
    IASSERT(V->lens[i] == mat_list[i]->nrow);
  }

  this->V = V;
  this->W = mat_list;
  this->mttkrp = (Matrix<> **)malloc(V->order * sizeof(Matrix<> *));
  this->mttkrp_local_mat = (Matrix<> **)malloc(V->order * sizeof(Matrix<> *));

  this->arrs = (dtype **)malloc(sizeof(dtype *) * V->order);
  this->arrs_mttkrp = (dtype **)malloc(sizeof(dtype *) * V->order);

  this->phys_phase = (int *)malloc(order * sizeof(int));
  for (int i = 0; i < order; i++) {
    this->phys_phase[i] = V->edge_map[i].calc_phys_phase();
  }

  // W_local
  this->W_local = (Matrix<> **)malloc(V->order * sizeof(Matrix<> *));
  for (int i = 0; i < this->order; i++) {
    int64_t pad_local_col = int(this->V->pad_edge_len[i] / this->phys_phase[i]);
    this->W_local[i] = new Matrix<dtype>(pad_local_col, this->rank, *sworld);
  }

  this->ldas = (int64_t *)malloc(order * sizeof(int64_t));
  ldas[0] = 1;
  for (int i = 1; i < this->order; i++) {
    ldas[i] = ldas[i - 1] * V->lens[i - 1];
  }

  this->par = Partition(V->topo->order, V->topo->lens);
  this->par_idx = (char *)malloc(sizeof(char) * V->topo->order);
  for (int i = 0; i < V->topo->order; i++) {
    par_idx[i] = 'a' + i + 1;
  }

  // W_remap
  // Note: W_remap only serves as an data transfer intermediate. It shouldn't
  // take part in any calculations.
  this->W_remap = (Matrix<> **)malloc(V->order * sizeof(Matrix<> *));
  construct_W_remap(this->W, this->W_remap);

  // cm_reduce
  this->cm_reduce = (MPI_Comm **)malloc(V->order * sizeof(MPI_Comm *));
  this->cmr_reduce = (int **)malloc(V->order * sizeof(int *));
}


#include <ctf.hpp>

using namespace CTF;

template <typename dtype>
LocalMTTKRP<dtype>::LocalMTTKRP(int order, int r, World &dw) {
  this->world = &dw;
  this->order = order;
  this->rank = r;
}

template <typename dtype> LocalMTTKRP<dtype>::~LocalMTTKRP() {
  free(redist_mats);
  free(par_idx);
  free(phys_phase);
  free(ldas);
  free(arrs);
  if (!V->is_sparse)
    V->sr->pair_dealloc((char *)pairs);
}

template <typename dtype> void LocalMTTKRP<dtype>::distribute_mats(int mode) {

  IASSERT(mode >= 0 && mode < this->order);

  Timer t_mttkrp_remap("MTTKRP_remap_mats");
  t_mttkrp_remap.start();

  for (int i = 0; i < order; i++) {
    cout << "i is: " << i << endl;
    Tensor<dtype> *mat = &(this->W[i]);

    if (this->phys_phase[i] == 1) {
      cout << "phys phase is 1: " << endl;
      this->redist_mats[i] = NULL;
      if (this->world->np == 1) {
        this->arrs[i] = (dtype *)this->W[i].data;
        /*
        if (i == mode)
          std::fill(arrs[i], arrs[i] + mat_list[i].size,
                    *((dtype *)V->sr->addid()));
        */
      } else if (i != mode) {
        this->arrs[i] = (dtype *)V->sr->alloc(V->lens[i] * this->rank);
        mat->read_all(arrs[i], true);
      } else {
        char nonastr[2];
        nonastr[0] = 'a' - 1;
        nonastr[1] = 'a' - 2;
        this->redist_mats[i] = new Matrix<dtype>(
            this->W[i].nrow, this->rank, nonastr, par[par_idx], Idx_Partition(),
            0, *this->world, *V->sr);
        // TODO: this is for debug
        this->redist_mats[i]->operator[]("ij") = this->W[i]["ij"];
        arrs[i] = (dtype *)this->redist_mats[i]->data;
      }

    } else {

      cout << "phys phase is  not 1: " << endl;

      int topo_dim = V->edge_map[i].cdt;
      IASSERT(V->edge_map[i].type == CTF_int::PHYSICAL_MAP);
      IASSERT(!V->edge_map[i].has_child ||
              V->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);

      char mat_idx[2];
      mat_idx[0] = par_idx[topo_dim];
      mat_idx[1] = 'a';

      int comm_lda = 1;
      for (int l = 0; l < topo_dim; l++) {
        comm_lda *= V->topo->dim_comm[l].np;
      }
      CTF_int::CommData cmdt(V->wrld->rank -
                                 comm_lda * V->topo->dim_comm[topo_dim].rank,
                             V->topo->dim_comm[topo_dim].rank, V->wrld->cdt);

      Matrix<dtype> *m =
          new Matrix<dtype>(this->W[i].nrow, this->rank, mat_idx, par[par_idx],
                            Idx_Partition(), 0, *V->wrld, *V->sr);
      // if (i != mode)
      m->operator[]("ij") = mat->operator[]("ij");
      cout << "print m here:  " << i << endl;
      m->print();
      cout << "print mat_list here:  " << i << endl;
      this->W[i].print();

      this->redist_mats[i] = m;
      cout << "print redist_mats here:  " << i << endl;
      this->redist_mats[i]->print();

      // TODO:change here
      // redist_mats[i]->operator[]("ij") = mat->operator[]("ij");
      arrs[i] = (dtype *)m->data;

      // if (i != mode)
      cmdt.bcast(m->data, m->size, V->sr->mdtype(), 0);
    }
  }
  t_mttkrp_remap.stop();
}

template <typename dtype>
void LocalMTTKRP<dtype>::post_mttkrp_reduce(int mode) {
  for (int j = 0; j < V->order; j++) {
    if (j == mode) {
      int red_len = this->world->np / this->phys_phase[j];
      // if (red_len > 1) {
      //   int64_t sz;
      //   if (redist_mats[j] == NULL) {
      //     sz = T->lens[j] * kd;
      //   } else {
      //     sz = redist_mats[j]->size;
      //   }
      //   int jr = T->edge_map[j].calc_phys_rank(T->topo);
      //   MPI_Comm cm;
      //   MPI_Comm_split(T->wrld->comm, jr, T->wrld->rank, &cm);
      //   int cmr;
      //   MPI_Comm_rank(cm, &cmr);

      //   Timer t_mttkrp_red("MTTKRP_Reduce");
      //   t_mttkrp_red.start();
      //   if (cmr == 0)
      //     MPI_Reduce(MPI_IN_PLACE, arrs[j], sz, T->sr->mdtype(),
      //                T->sr->addmop(), 0, cm);
      //   else {
      //     MPI_Reduce(arrs[j], NULL, sz, T->sr->mdtype(), T->sr->addmop(),
      //     0,
      //                cm);
      //     std::fill(arrs[j], arrs[j] + sz, *((dtype *)T->sr->addid()));
      //   }
      //   t_mttkrp_red.stop();
      //   MPI_Comm_free(&cm);
      // }
      if (this->redist_mats[j] != NULL) {
        // TODO: this is for debug
        // cout << "print redist mat here:  " << endl;
        // redist_mats[j]->print();
        // mat_list[j].print();
        this->W[j].set_zero();
        this->W[j].operator[]("ij") += this->redist_mats[j]->operator[]("ij");
        delete redist_mats[j];
      } else {
        IASSERT((dtype *)this->W[j].data == arrs[j]);
      }
    } else {
      if (redist_mats[j] != NULL) {
        if (redist_mats[j]->data != (char *)arrs[j])
          V->sr->dealloc((char *)arrs[j]);
        delete redist_mats[j];
      } else {
        if (arrs[j] != (dtype *)this->W[j].data)
          V->sr->dealloc((char *)arrs[j]);
      }
    }
  }
}

template <typename dtype>
void LocalMTTKRP<dtype>::setup(Tensor<dtype> *V, Matrix<dtype> *mat_list) {

  IASSERT(mat_list[0].ncol == this->rank);
  IASSERT(this->order == V->order);
  for (int i = 0; i < this->order; i++) {
    IASSERT(V->lens[i] == mat_list[i].nrow);
  }

  this->V = V;
  this->W = mat_list;
  this->arrs = (dtype **)malloc(sizeof(dtype *) * V->order);

  this->phys_phase = (int *)malloc(order * sizeof(int));
  for (int i = 0; i < order; i++) {
    this->phys_phase[i] = V->edge_map[i].calc_phys_phase();
  }

  V->get_local_pairs(&this->npair, &this->pairs, true, false);

  this->ldas = (int64_t *)malloc(order * sizeof(int64_t));
  ldas[0] = 1;
  for (int i = 1; i < this->order; i++) {
    ldas[i] = ldas[i - 1] * V->lens[i - 1];
  }

  this->redist_mats = (Tensor<dtype> **)malloc(sizeof(Tensor<dtype> *) * order);

  this->par = Partition(V->topo->order, V->topo->lens);
  this->par_idx = (char *)malloc(sizeof(char) * V->topo->order);
  for (int i = 0; i < V->topo->order; i++) {
    par_idx[i] = 'a' + i + 1;
  }
}

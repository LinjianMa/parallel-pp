
#include <ctf.hpp>

template <typename dtype>
void MTTKRP1(Tensor<dtype> *T, int order, int64_t *lens, int *phys_phase,
             int64_t k, int64_t nnz, int out_mode,
             CTF::Pair<dtype> const *tsr_data, dtype const *const *op_mats,
             dtype *out_mat) {
  // if (aux_mode_first){
  dtype *buffer = (dtype *)((Semiring<dtype> *)T->sr)->alloc(k);
  dtype *out_buffer;
  if (out_mode != 0)
    out_buffer = (dtype *)((Semiring<dtype> *)T->sr)->alloc(k);
  int64_t *inds = (int64_t *)malloc(sizeof(int64_t) * (order - 1));
  int64_t idx = 0;
  while (idx < nnz) {
    int64_t fiber_idx = tsr_data[idx].k / lens[0];
    int64_t fi = fiber_idx;
    for (int i = 0; i < order - 1; i++) {
      inds[i] = (fi % lens[i + 1]) / phys_phase[i + 1];
      fi = fi / lens[i + 1];
    }
    int64_t fiber_nnz = 1;
    while (idx + fiber_nnz < nnz &&
           tsr_data[idx + fiber_nnz].k / lens[0] == fiber_idx)
      fiber_nnz++;
    if (out_mode == 0) {
      memcpy(buffer, op_mats[1] + inds[0] * k, k * sizeof(dtype));
      for (int i = 1; i < order - 1; i++) {
        ((Semiring<dtype> *)T->sr)
            ->fvmul(buffer, op_mats[i + 1] + inds[i] * k, buffer, k);
      }
      for (int64_t i = idx; i < idx + fiber_nnz; i++) {
        int64_t kk = (tsr_data[i].k % lens[0]) / phys_phase[0];
        ((Semiring<dtype> *)T->sr)
            ->faxpy(k, tsr_data[i].d, buffer, 1, out_mat + kk * k, 1);
      }
    } else {
      int64_t ok = inds[out_mode - 1];
      if (out_mode > 1)
        memcpy(buffer, op_mats[1] + inds[0] * k, k * sizeof(dtype));
      else if (order > 2)
        memcpy(buffer, op_mats[2] + inds[1] * k, k * sizeof(dtype));
      else
        std::fill(buffer, buffer + k, ((Semiring<dtype> *)T->sr)->tmulid);
      for (int i = 1 + (out_mode == 1); i < order - 1; i++) {
        if (out_mode != i + 1)
          ((Semiring<dtype> *)T->sr)
              ->fvmul(buffer, op_mats[i + 1] + inds[i] * k, buffer, k);
      }
      std::fill(out_buffer, out_buffer + k, ((Semiring<dtype> *)T->sr)->taddid);
      for (int64_t i = idx; i < idx + fiber_nnz; i++) {
        int64_t kk = (tsr_data[i].k % lens[0]) / phys_phase[0];
        ((Semiring<dtype> *)T->sr)
            ->faxpy(k, tsr_data[i].d, op_mats[0] + kk * k, 1, out_buffer, 1);
      }
      ((Semiring<dtype> *)T->sr)->fvmul(out_buffer, buffer, out_buffer, k);
      ((Semiring<dtype> *)T->sr)
          ->faxpy(k, ((Semiring<dtype> *)T->sr)->tmulid, out_buffer, 1,
                  out_mat + ok * k, 1);
      // for (int j=0; j<k; j++){
      //  out_mat[j+ok*k] += out_buffer[j]*buffer[j];
      //}
    }
    idx += fiber_nnz;
  }
  if (out_mode != 0)
    ((Semiring<dtype> *)T->sr)->dealloc((char *)out_buffer);
  ((Semiring<dtype> *)T->sr)->dealloc((char *)buffer);
  free(inds);
  // }
  // else {
  //   IASSERT(0);
  // }
}

template <typename dtype>
void MTTKRP(Tensor<dtype> *T, Matrix<dtype> *mat_list, int mode) {

  Timer t_mttkrp("MTTKRP");
  t_mttkrp.start();
  int k = mat_list[0].ncol;

  IASSERT(mode >= 0 && mode < T->order);
  for (int i = 0; i < T->order; i++) {
    IASSERT(T->lens[i] == mat_list[i].nrow);
  }
  dtype **arrs = (dtype **)malloc(sizeof(dtype *) * T->order);
  int64_t *ldas = (int64_t *)malloc(T->order * sizeof(int64_t));
  int *phys_phase = (int *)malloc(T->order * sizeof(int));
  int *mat_strides = NULL;
  mat_strides = (int *)malloc(2 * T->order * sizeof(int));
  for (int i = 0; i < T->order; i++) {
    phys_phase[i] = T->edge_map[i].calc_phys_phase();
  }

  int64_t npair;
  Pair<dtype> *pairs;
  T->get_local_pairs(&npair, &pairs, true, false);

  ldas[0] = 1;
  for (int i = 1; i < T->order; i++) {
    ldas[i] = ldas[i - 1] * T->lens[i - 1];
  }

  Tensor<dtype> **redist_mats =
      (Tensor<dtype> **)malloc(sizeof(Tensor<dtype> *) * T->order);
  Partition par(T->topo->order, T->topo->lens);
  char *par_idx = (char *)malloc(sizeof(char) * T->topo->order);
  for (int i = 0; i < T->topo->order; i++) {
    par_idx[i] = 'a' + i + 1;
  }
  char mat_idx[2];
  int slice_st[2];
  int slice_end[2];
  int k_start = 0;
  int kd = 0;
  int div = 1;

  for (int d = 0; d < div; d++) {
    k_start += kd;
    kd = k / div + (d < k % div);
    int k_end = k_start + kd;

    Timer t_mttkrp_remap("MTTKRP_remap_mats");
    t_mttkrp_remap.start();
    for (int i = 0; i < T->order; i++) {
      cout << "i is: " << i << endl;
      Tensor<dtype> mmat;
      Tensor<dtype> *mat = &mat_list[i];

      int64_t tot_sz;

      tot_sz = T->lens[i] * kd;

      mat_strides[2 * i + 0] = 1;
      mat_strides[2 * i + 1] = T->lens[i];

      int nrow, ncol;

      nrow = T->lens[i];
      ncol = kd;

      if (phys_phase[i] == 1) {
        cout << "phys phase is 1: " << endl;
        redist_mats[i] = NULL;
        if (T->wrld->np == 1) {
          IASSERT(div == 1);
          arrs[i] = (dtype *)mat_list[i].data;
          /*
          if (i == mode)
            std::fill(arrs[i], arrs[i] + mat_list[i].size,
                      *((dtype *)T->sr->addid()));
          */
        } else if (i != mode) {
          arrs[i] = (dtype *)T->sr->alloc(tot_sz);
          mat->read_all(arrs[i], true);
        } else {

          char nonastr[2];
          nonastr[0] = 'a' - 1;
          nonastr[1] = 'a' - 2;
          redist_mats[i] =
              new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx],
                                Idx_Partition(), 0, *T->wrld, *T->sr);
          // TODO: this is for debug
          redist_mats[i]->operator[]("ij") = mat_list[i]["ij"];
          arrs[i] = (dtype *)redist_mats[i]->data;
        }

      } else {

        cout << "phys phase is  not 1: " << endl;

        int topo_dim = T->edge_map[i].cdt;
        IASSERT(T->edge_map[i].type == CTF_int::PHYSICAL_MAP);
        IASSERT(!T->edge_map[i].has_child ||
                T->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);

        mat_idx[0] = par_idx[topo_dim];
        mat_idx[1] = 'a';

        int comm_lda = 1;
        for (int l = 0; l < topo_dim; l++) {
          comm_lda *= T->topo->dim_comm[l].np;
        }
        CTF_int::CommData cmdt(T->wrld->rank -
                                   comm_lda * T->topo->dim_comm[topo_dim].rank,
                               T->topo->dim_comm[topo_dim].rank, T->wrld->cdt);

        Matrix<dtype> *m =
            new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx],
                              Idx_Partition(), 0, *T->wrld, *T->sr);
        // if (i != mode)
        m->operator[]("ij") = mat->operator[]("ij");
        cout << "print m here:  " << i << endl;
        m->print();
        cout << "print mat_list here:  " << i << endl;
        mat_list[i].print();

        redist_mats[i] = m;
        cout << "print redist_mats here:  " << i << endl;
        redist_mats[i]->print();

        // TODO:change here
        // redist_mats[i]->operator[]("ij") = mat->operator[]("ij");
        arrs[i] = (dtype *)m->data;

        // if (i != mode)
        cmdt.bcast(m->data, m->size, T->sr->mdtype(), 0);

        mat_strides[2 * i + 0] = 1;
        mat_strides[2 * i + 1] = m->pad_edge_len[0] / phys_phase[i];
      }
    }
    t_mttkrp_remap.stop();

    Timer t_mttkrp_work("MTTKRP_work");
    t_mttkrp_work.start();
    {
      // MTTKRP1(T, T->order, T->lens, phys_phase, kd, npair, mode, pairs, arrs,
      // arrs[mode]);
    }
    t_mttkrp_work.stop();

    for (int j = 0; j < T->order; j++) {
      if (j == mode) {
        int red_len = T->wrld->np / phys_phase[j];
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
        if (redist_mats[j] != NULL) {
          // TODO: this is for debug
          // cout << "print redist mat here:  " << endl;
          // redist_mats[j]->print();
          // mat_list[j].print();
          mat_list[j].set_zero();
          mat_list[j].operator[]("ij") += redist_mats[j]->operator[]("ij");
          delete redist_mats[j];
        } else {
          IASSERT((dtype *)mat_list[j].data == arrs[j]);
        }
      } else {
        if (redist_mats[j] != NULL) {
          if (redist_mats[j]->data != (char *)arrs[j])
            T->sr->dealloc((char *)arrs[j]);
          delete redist_mats[j];
        } else {
          if (arrs[j] != (dtype *)mat_list[j].data)
            T->sr->dealloc((char *)arrs[j]);
        }
      }
    }
  }
  free(redist_mats);
  if (mat_strides != NULL)
    free(mat_strides);
  free(par_idx);
  free(phys_phase);
  free(ldas);
  free(arrs);
  if (!T->is_sparse)
    T->sr->pair_dealloc((char *)pairs);
  t_mttkrp.stop();
}

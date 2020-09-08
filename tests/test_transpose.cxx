#include <ctf.hpp>

using namespace CTF;

void TEST_transpose(World &dw) {
  if (dw.rank == 0) {
    cout << "Test transpose" << endl;
  }

  // test init
  int lens[3];
  int lens_out[3];
  int size = 200;
  int rank = 200;
  for (int i = 0; i < 3; i++) {
    lens[i] = size;
    lens_out[i] = size;
  }
  lens_out[2] = rank;

  Tensor<> V = Tensor<>(3, lens, dw);
  V.fill_random(0, 1);
  Matrix<> W = Matrix<>(size, 200, dw);
  W.fill_random(0, 1);
  Tensor<> V_out = Tensor<>(3, lens_out, dw);

  Timer_epoch t_mttkrp_first_contract("mttkrp_first_contract");
  t_mttkrp_first_contract.begin();
  V_out["car"] += V["abc"] * W["br"];
  t_mttkrp_first_contract.end();
  
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  World dw(argc, argv);

  TEST_transpose(dw);

  if (dw.rank == 0) {
    cout << "All tests passed" << endl;
  }
  MPI_Finalize();
}
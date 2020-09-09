#include <ctf.hpp>

using namespace CTF;

void TEST_transpose(World &dw) {
  if (dw.rank == 0) {
    cout << "Test transpose" << endl;
  }

  // test init
  int lens[3];
  int lens_out[3];
  int size = 50;
  int rank = 50;
  for (int i = 0; i < 3; i++) {
    lens[i] = size;
    lens_out[i] = size;
  }
  lens_out[2] = rank;

  Tensor<> V = Tensor<>(3, lens, dw);
  V.fill_random(0, 1);
  Matrix<> W = Matrix<>(size, rank, dw);
  W.fill_random(0, 1);
  Tensor<> V_out = Tensor<>(3, lens_out, dw);

  Timer_epoch t_mttkrp_ttm("mttkrp_ttm");
  t_mttkrp_ttm.begin();
  V_out["car"] += V["abc"] * W["br"];
  t_mttkrp_ttm.end();

  Timer_epoch t_mttkrp_ttv("mttkrp_ttv");
  int lens2[4] = {size, size, size, rank};
  Tensor<> V2 = Tensor<>(4, lens2, dw);
  V2.fill_random(0, 1);
  t_mttkrp_ttv.begin();
  V_out["abr"] += V2["abcr"] * W["cr"];
  t_mttkrp_ttv.end();
  
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
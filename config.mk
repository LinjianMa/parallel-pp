#main compiler to use
CXX=mpicxx
#compiler flags (-g -O0 for debug, -O3 for optimization), generally need -fopenmp and -std=c++0x
# [On Mac] CXXFLAGS= -O0 -std=c++0x -DOMP_OFF -Wall -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -D_DARWIN_C_SOURCE -DFTN_UNDERSCORE=1 -DUSE_LAPACK -DUSE_SCALAPACK 
CXXFLAGS=-std=c++0x -O0  -fopenmp -no-ipo
#path to CTF include directory prefixed with -I
# [On Mac] INCLUDE_PATH=-I/Users/linjian/Documents/ctf/include
INCLUDE_PATH=-I/work/03940/lma16/stampede2/ctf-conda/include
#path to MPI/CTF/scalapack/lapack/blas library directories prefixed with -L
# [On Mac] LIB_PATH= -L/Users/linjian/Documents/ctf/scalapack/build/lib  -L/Users/linjian/Documents/ctf/lib -L/Users/linjian/Documents/ctf/lib_shared
LIB_PATH=-L/home1/03940/lma16/anaconda3/lib  -L/work/03940/lma16/stampede2/ctf-conda/lib -L/work/03940/lma16/stampede2/ctf-conda/hptt/lib
#libraries to link (MPI/CTF/scalapack/lapack/blas) to prefixed with -l
# [On Mac] LIBS=-lctf -lscalapack -llapack -lblas
LIBS=-lctf -lmkl_scalapack_lp64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -Wl,--end-group -lpthread -lm -Wl,-Bstatic -lhptt -Wl,-Bdynamic

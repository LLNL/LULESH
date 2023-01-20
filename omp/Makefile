# Default build suggestion of MPI + OPENMP with Clang on IBM (Power 8) + NVIDIA GPU machines.
# You might have to change the compiler name and flags.

SHELL = /bin/sh
.SUFFIXES: .cc .o

LULESH_EXEC = lulesh2.0

MPI_INC = /opt/local/include/openmpi
MPI_LIB = /opt/local/lib

# Point your mpicc to Clang
CXX ?= mpicc

SOURCES2.0 = \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc
OBJECTS2.0 = $(SOURCES2.0:.cc=.o)

teams =
ifdef TEAMS
teams = -DTEAMS=$(TEAMS)
endif

threads =
ifdef THREADS
threads = -DTHREADS=$(THREADS)
endif

mpi=-DUSE_MPI=0
gpu=-DUSE_GPU=1
ifdef USE_GPU
gpu = -DUSE_GPU=$(USE_GPU)
endif

# Tuning flags for Power 8
CXXFLAGS= -O3 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_70 -fopenmp-offload-mandatory -foffload-lto $(shared) $(mpi) $(teams) $(threads) $(gpu)

.cc.o: lulesh.h
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

all: $(LULESH_EXEC)

lulesh2.0: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(CXXFLAGS) $(OBJECTS2.0) $(LDFLAGS) -lm -o $@

clean:
	/bin/rm -f *.o *~ *.tgt* $(OBJECTS) $(LULESH_EXEC)
	/bin/rm -rf *.dSYM

tar: clean
	cd .. ; tar cvf lulesh-2.0.tar LULESH-2.0 ; mv lulesh-2.0.tar LULESH-2.0


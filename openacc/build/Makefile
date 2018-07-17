SHELL = /bin/sh
.SUFFIXES: .cc .o

LULESH_EXEC = lulesh2.0

MPI_INC = /usr/local/include/openmpi
MPI_LIB = /usr/local/lib

SERCXX = pgCC -DUSE_MPI=0
MPICXX = mpicxx -DUSE_MPI=1
CXX = $(MPICXX)

SOURCES2.0 = \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-util.cc \
	lulesh-init.cc \

OBJECTS2.0 = $(SOURCES2.0:.cc=.o)

#CXXFLAGS = -O3

#SILO_INCDIR = /opt/local/include
#SILO_LIBDIR = /opt/local/lib
#SILO_INCDIR = /usr/gapps/silo/current/chaos_5_x86_64/include
#SILO_LIBDIR = /usr/gapps/silo/current/chaos_5_x86_64/lib
SILO_INCDIR = /usr/gapps/silo/4.8/chaos_5_x86_64_ib_pgcc/include
SILO_LIBDIR = /usr/gapps/silo/4.8/chaos_5_x86_64_ib_pgcc/lib

ACCFLAGS = -acc -ta=nvidia,cc35 -Minfo=accel
CXXFLAGS = -mp --restrict -Mautoinline -Minline=levels:20 $(ACCFLAGS) -O2 
LDFLAGS = -mp $(ACCFLAGS) 

.cc.o: lulesh.h
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@ $<

all: $(LULESH_EXEC)

lulesh2.0: $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -lm -o $@

clean:
	/bin/rm -f *.gpu *.o *~ $(OBJECTS) $(LULESH_EXEC)
	/bin/rm -rf *.dSYM

tar: clean
	cd .. ; tar cvf lulesh-2.0.tar LULESH-2.0 ; mv lulesh-2.0.tar LULESH-2.0


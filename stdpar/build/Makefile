# Build with nvc++, with parallel algorithm support turned on.

SHELL = /bin/sh
.SUFFIXES: .cc .o

LULESH_EXEC = lulesh2.0

CXX = nvc++

SOURCES2.0 = \
	lulesh.cc \
	lulesh-comm.cc \
	lulesh-viz.cc \
	lulesh-util.cc \
	lulesh-init.cc
OBJECTS2.0 = $(SOURCES2.0:.cc=.o)

CXXFLAGS = -w -fast -Mnouniform -Mfprelaxed -stdpar=gpu -std=c++11 -DUSE_MPI=0
LDFLAGS = $(CXXFLAGS) 

%.o: ../src/%.cc ../src/lulesh.h
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

all: $(LULESH_EXEC)

$(LULESH_EXEC): $(OBJECTS2.0)
	@echo "Linking"
	$(CXX) $(OBJECTS2.0) $(LDFLAGS) -lm -o $@

run: $(LULESH_EXEC)
	./lulesh2.0 -p -s 150

clean:
	/bin/rm -f *.o *~ $(OBJECTS) $(LULESH_EXEC)
	/bin/rm -rf *.dSYM


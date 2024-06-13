# Here is an example Makefile

CXXFLAGS += -O3 -march=native -DNDEBUG -DARMA_NO_DEBUG -fopenmp

# Include armadillo and liblbfgs paths here
# CXXFLAGS += -I/path/to/dependencies
# CXXFLAGS += -L/path/to/dependencies

LDFLAGS += -lopenblas

# These may be necessary
LDFLAGS += -lquadmath
LDFLAGS += -llbfgs

all: single mpi

single: sweep_lr sweep_naive_avg sweep_owa sweep_csl sweep_dane sweep_prox_csl
mpi: sweep_mpi_naive_avg sweep_mpi_owa sweep_mpi_csl sweep_mpi_dane sweep_mpi_pcsl

# Single-node targets
sweep_lr: src/sweep_lr.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_lr.o $(LDFLAGS)

sweep_naive_avg: src/sweep_naive_avg.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_naive_avg.o $(LDFLAGS)

sweep_owa: src/sweep_owa.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_owa.o $(LDFLAGS)

sweep_csl: src/sweep_csl.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_csl.o $(LDFLAGS)

sweep_dane: src/sweep_dane.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_dane.o $(LDFLAGS)

sweep_prox_csl: src/sweep_prox_csl.o
	$(CXX) $(CXXFLAGS) -o $@ src/sweep_prox_csl.o $(LDFLAGS)

# MPI targets
sweep_mpi_naive_avg:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_naive_avg.cpp $(LDFLAGS)

sweep_mpi_owa:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_owa.cpp $(LDFLAGS)

sweep_mpi_csl:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_csl.cpp $(LDFLAGS)

sweep_mpi_dane:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_dane.cpp $(LDFLAGS)

sweep_mpi_pcsl:
	mpicxx $(CXXFLAGS) -o $@ src/mpi/sweep_mpi_pcsl.cpp $(LDFLAGS)

clean:
	rm -f src/*.o sweep_*

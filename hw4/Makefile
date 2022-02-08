SHELL:=/bin/bash

CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: prep homework4

prep:
	. /usr/share/Modules/init/bash; \
        module add openmpi/4.0.5 \
		module add opencv/3.2.0-openmpi

parallel = ${MPI_THREADS}

run: run_opening run_hw4
	@echo " "; echo "Runs completed"

run_opening:
	@echo "Running tests with ${parallel} threads";

run_hw4: homework4
	@echo "Running the simple mpi test" ;\
	time mpirun -np ${parallel} homework4 ./Astronaught.png 4 256 output;

# Test Program
homework4: main.o
	mpic++ $(CFLAGS) $(LIBS) -o homework4 main.o

main.o:
	mpic++ -c main.cpp

clean:
	rm -f homework4 *.o 


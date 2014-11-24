#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]){
	std::cout<<"Hello Worlds!"<<std::endl;

	int rank, size;
	MPI_Init(&argc, &argv); //Start up MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); //Get current processor number
	MPI_Comm_size(MPI_COMM_WORLD, &size); //Get number of processors in communicator
	std::cout<<"Process "<<rank<<" of "<<size<<" online"<<std::endl;

	MPI_Finalize();

	return 0;
}

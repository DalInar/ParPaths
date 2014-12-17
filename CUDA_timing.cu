#include <iostream>
#include <sstream>
#include <set>

__global__ void empty_kernel(){

}

int main(){

	int * data;
	int * d_data;

	std::ofstream output;
	output.open("CUDA_kernel_timing.txt");

	int num_blocks;
	int num_threads;


}

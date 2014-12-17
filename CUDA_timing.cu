#include <iostream>
#include <sstream>
#include <set>
#include "CSR_Graph.h"

__global__ void empty_kernel(){

}

int main(){

	int * data;
	int * d_data;

	std::ofstream output;
	output.open("CUDA_kernel_timing.txt");

	int num_blocks;
	int num_threads;

	int trials = 10000;

	boost::timer::cpu_timer timer;
	for(int i=0; i<trials; i++){
		empty_kernel<<<1, 1024>>>();
		cudaDeviceSynchronize();
	}
	timer.stop();

	output<<"10000 kernel launches, <<<1,1024>>>, average time = "<<( (double) timer.elapsed().wall / 1000000000.0 ) / trials<< std::endl;

	timer.start();
	for(int i=0; i<trials; i++){
		empty_kernel<<<1024, 1>>>();
		cudaDeviceSynchronize();
	}
	timer.stop();

	output<<"10000 kernel launches, <<<1024,1>>>, average time = "<< ( (double) timer.elapsed().wall / 1000000000.0 ) / trials<<std::endl;

	timer.start();
	for(int i=0; i<trials; i++){
		empty_kernel<<<64, 64>>>();
		cudaDeviceSynchronize();
	}
	timer.stop();

	output<<"10000 kernel launches, <<<64,64>>>, average time = "<< ( (double) timer.elapsed().wall / 1000000000.0 ) / trials<<std::endl;



}

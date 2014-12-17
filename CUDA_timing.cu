#include <iostream>
#include <sstream>
#include <set>
#include "CSR_Graph.h"

__global__ void empty_kernel(){

}

int main(){

	double * data;
	double * d_data;

	std::ofstream output;
	output.open("CUDA_kernel_timing.txt");

	int num_blocks;
	int num_threads;

	int trials = 100000;

	boost::timer::cpu_timer timer;
	for(int i=0; i<trials; i++){
		empty_kernel<<<1, 1024>>>();
		cudaDeviceSynchronize();
	}
	timer.stop();

	output<<trials<<" kernel launches, <<<1,1024>>>, average time = "<<( (double) timer.elapsed().wall / 1000000000.0 ) / trials<< std::endl;

	timer.start();
	for(int i=0; i<trials; i++){
		empty_kernel<<<1024, 1>>>();
		cudaDeviceSynchronize();
	}
	timer.stop();

	output<<trials<<" kernel launches, <<<1024,1>>>, average time = "<< ( (double) timer.elapsed().wall / 1000000000.0 ) / trials<<std::endl;

	timer.start();
	for(int i=0; i<trials; i++){
		empty_kernel<<<64, 64>>>();
		cudaDeviceSynchronize();
	}
	timer.stop();

	output<<trials<<" kernel launches, <<<64,64>>>, average time = "<< ( (double) timer.elapsed().wall / 1000000000.0 ) / trials<<std::endl;

	std::vector <int> num_doubles;
	num_doubles.push_back(100);
	num_doubles.push_back(1000);
	num_doubles.push_back(10000);
	num_doubles.push_back(100000);
	num_doubles.push_back(1000000);
	num_doubles.push_back(10000000);

	trials = 100;
	int data_size;
	for(int i=0; i<num_doubles.size(); i++){
		data_size = num_doubles[i] * sizeof(double);
		data = (double *) malloc(data_size);
		cudaMalloc((void **) & d_data, data_size);

		timer.start();
		for(int j=0; j< trials; j++){
			cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
			cudaMemcpy(data, d_data, data_size, cudaMemcpyDeviceToHost);
		}
		timer.stop();

		output<<trials<<" memory writes and reads to GPU of size "<<data_size<<", average time = "<< ( (double) timer.elapsed().wall / 1000000000.0 ) / trials<<std::endl;

	}


	return 0;
}

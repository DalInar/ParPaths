#include <iostream>
#include <sstream>
#include <set>
#include "CSR_Graph.h"

__global__ void empty_kernel(){

}

__global__ void global_memory_test(int size, int trials, int * data){
	int ID = blockIdx.x *blockDim.x + threadIdx.x;


	if(ID < size){
		int i;
		int d;
		int target;
		int over;
		for(i=0; i<trials; i++){
			target = threadIdx.x + i;
			over = (target/blockDim.x);
			target = blockIdx.x*blockDim.x + target - over*blockDim.x;
			d = data[target];
			d += i;
			data[target] = d;
		}
	}
}

__global__ void shared_memory_test(int size, int trials, int * data){
	int GLOBAL_ID = blockIdx.x *blockDim.x + threadIdx.x;
	int LOCAL_ID = threadIdx.x;
	extern __shared__ int s_data[];


	if(GLOBAL_ID < size){
		int target;
		int over;
		int i;
		int d;

		s_data[LOCAL_ID] = data[GLOBAL_ID];
		__syncthreads();

		for(i=0; i<trials; i++){
			target = threadIdx.x + i;
			over = (target/blockDim.x);
			target = target - over*blockDim.x;
			d = s_data[target];
			d += i;
			s_data[target] = d;
		}

		data[GLOBAL_ID] = s_data[LOCAL_ID];
	}
}

int main(){

	double * data;
	double * d_data;

	std::ofstream output;
	output.open("CUDA_kernel_timing.txt");

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
	num_doubles.push_back(100000000);
	num_doubles.push_back(1000000000);

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

	int num_access = 100000000;
	int n_ints=2688;
	int mem_data_size = n_ints*sizeof(int);
	int * mem_data;
	int * d_mem_data;

	int num_threads_per_block = 32;
	int num_blocks = n_ints / num_threads_per_block;

	mem_data = (int *) malloc(mem_data_size);
	cudaMalloc((void **) & d_mem_data, mem_data_size);

	for(int i=0; i<n_ints; i++){
		mem_data[i] = i;
	}

	cudaMemcpy(d_mem_data, mem_data, mem_data_size, cudaMemcpyHostToDevice);

	timer.start();
	global_memory_test<<<num_blocks, num_threads_per_block>>>(n_ints, num_access, d_mem_data);
	cudaDeviceSynchronize();
	timer.stop();

	output<<num_access<<" global memory writes and reads on GPU, average time per read/write= "<< ( (double) timer.elapsed().wall / 1000000000.0 ) / num_access<<std::endl;

	cudaMemcpy(d_mem_data, mem_data, mem_data_size, cudaMemcpyHostToDevice);

	timer.start();
	shared_memory_test<<<num_blocks, num_threads_per_block, num_threads_per_block*sizeof(int)>>>(n_ints, num_access, d_mem_data);
	cudaDeviceSynchronize();
	timer.stop();

	output<<num_access<<" shared memory writes and reads on GPU, average time per read/write= "<< ( (double) timer.elapsed().wall / 1000000000.0 ) / num_access<<std::endl;




	return 0;
}

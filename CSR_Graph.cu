/*
 * CSR_Graph.cu
 *
 *  Created on: Dec 12, 2014
 *      Author: pakij
 */

#include "CSR_Graph.h"

__global__ void BellmanFord_cuda(int *offsets, int *edge_dests, double *weights){
	//weights[blockIdx.x] += 5.7;
	offsets[blockIdx.x] += 2;
	//edge_dests[blockIdx.x] += 1;
}

void CSR_Graph::BellmanFordGPU(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight){
	int num_threads = V;

	int *  d_offsets;
	int * d_edge_dests;
	double * d_weights;

	int * d_predecessors;
	double * d_path_weight;

	//Size of CSR graph
	int offsets_size = V*sizeof(int);
	int edge_dests_size = E*sizeof(int);
	int weights_size = E*sizeof(double);

	//Size of predecessor tree into
	int predecessors_size = V*sizeof(int);
	int path_weight_size = V*sizeof(double);

	//Allocate memory on device
	cudaMalloc((void **) & d_offsets, offsets_size);
	cudaMalloc((void **) & d_edge_dests, edge_dests_size);
	cudaMalloc((void **) & d_weights, weights_size);

	std::cout<<"Printing unmodified weights, offsets"<<std::endl;
	for(int i=0; i<V; i++){
		std::cout<<i<<" "<<weights[i]<<std::endl;
		std::cout<<i<<" "<<offsets[i]<<std::endl;
	}

	std::cout<<"Transferring to GPU"<<std::endl;
	cudaMemcpy(d_offsets, (int *) &offsets, offsets_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_dests, (int *) &edge_dests, edge_dests_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, (double *) &weights, weights_size, cudaMemcpyHostToDevice);

	std::cout<<"Running kernel"<<std::endl;
	BellmanFord_cuda<<<num_threads,1>>>(d_offsets,d_edge_dests,d_weights);

	//Copy results back to host
	cudaMemcpy((int *) &offsets, d_offsets, offsets_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((int *) &edge_dests, d_edge_dests, edge_dests_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((double *) &weights, d_weights, weights_size, cudaMemcpyDeviceToHost);

	std::cout<<"Printing GPU modified weights, offsets"<<std::endl;
	for(int i=0; i<V; i++){
		std::cout<<i<<" "<<weights[i]<<std::endl;
		std::cout<<i<<" "<<offsets[i]<<std::endl;
	}

	//cleanup
	cudaFree(d_offsets); cudaFree(d_edge_dests); cudaFree(d_weights);
}


//Simple test code
__global__ void test_add(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

bool CSR_Graph::test_cuda(){
	int N=1000;
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	bool result = true;

	int size=N*sizeof(int);
	a = (int *) malloc(size);
	b = (int *) malloc(size);
	c = (int *) malloc(size);
	cudaMalloc((void **) & d_a, size);
	cudaMalloc((void **) & d_b, size);
	cudaMalloc((void **) & d_c, size);

	std::cout<<std::endl<<"GPU output"<<std::endl;
	for(int i=0; i<N; i++){
		a[i]=i;
		b[i]=i*i;
	}

	//Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	test_add<<<N,1>>>(d_a,d_b,d_c);

	//Copy results back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);


	std::cout<<std::endl<<"GPU output"<<std::endl;
	for(int i=0; i<N; i++){
		//std::cout<<c[i]<<" ?= "<<a[i]+b[i]<<std::endl;
		if(c[i] != a[i] + b[i]){
			result = false;
		}
	}

	//cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	free(a); free(b); free(c);

	return result;
}

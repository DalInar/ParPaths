/*
 * CSR_Graph.cu
 *
 *  Created on: Dec 12, 2014
 *      Author: pakij
 */

#include "CSR_Graph.h"

__global__ void test_add(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void CSR_Graph::BellmanFordGPU(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight){

	int N=1000;
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

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
		std::cout<<c[i]<<" ?= "<<a[i]+b[i]<<std::endl;
	}

	//cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	free(a); free(b); free(c);
}




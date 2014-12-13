/*
 * CSR_Graph.cu
 *
 *  Created on: Dec 12, 2014
 *      Author: pakij
 */

#include "CSR_Graph.h"

__global__ void BellmanFord_cuda(int V, int E, int *offsets, int *edge_dests, double *weights, int * preds, double * path_weights){
	int my_vert = blockIdx.x;
	int source_vert;

	double my_dist = path_weights[my_vert];
	double trial_dist;

	source_vert=0;
	for(int i=0; i<E; i++){
		if(edge_dests[i] == my_vert){
			while(source_vert != V-1  && offsets[source_vert+1] < i){
				source_vert++;
			}
			trial_dist = weights[i] + path_weights[source_vert]; //Data race, possibly benign?
			//preds[blockIdx.x] = trial_dist;
			if(trial_dist < my_dist){
				path_weights[my_vert] = trial_dist;
				preds[my_vert] = source_vert;
			}
		}
	}
}

void CSR_Graph::BellmanFordGPU(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight){
	int num_threads = V;

	//Initialize predecessor tree
	predecessors.clear();
	path_weight.clear();
	double inf = std::numeric_limits<double>::infinity();
	predecessors.resize(V,-1);
	path_weight.resize(V,E*max_weight);
	predecessors[source_]=source_;
	path_weight[source_]=0;

	//GPU pointers
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
	cudaMalloc((void **) & d_predecessors, predecessors_size);
	cudaMalloc((void **) & d_path_weight, path_weight_size);

	std::cout<<"Transferring to GPU"<<std::endl;
	cudaMemcpy(d_offsets, (int *) &offsets[0], offsets_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_dests, (int *) &edge_dests[0], edge_dests_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, (double *) &weights[0], weights_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_predecessors, (int *) &predecessors[0], predecessors_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_path_weight, (double *) &path_weight[0], path_weight_size, cudaMemcpyHostToDevice);

	std::cout<<"Running kernel"<<std::endl;
	for(int iter=0; iter<1; iter++){
		std::cout<<iter<<std::endl;
		BellmanFord_cuda<<<num_threads,1>>>(V, E, d_offsets,d_edge_dests,d_weights,d_predecessors,d_path_weight);
		cudaDeviceSynchronize();
	}

	//Copy results back to host
	cudaMemcpy((int *) &offsets[0], d_offsets, offsets_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((int *) &edge_dests[0], d_edge_dests, edge_dests_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((double *) &weights[0], d_weights, weights_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((int *) &predecessors[0], d_predecessors, predecessors_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((double *) &path_weight[0], d_path_weight, path_weight_size, cudaMemcpyDeviceToHost);

	//cleanup
	cudaFree(d_offsets); cudaFree(d_edge_dests); cudaFree(d_weights);
	cudaFree(d_predecessors); cudaFree(d_path_weight);
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

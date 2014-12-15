/*
 * CSR_Graph.cu
 *
 *  Created on: Dec 12, 2014
 *      Author: pakij
 */

#include "CSR_Graph.h"

__device__ double atomicMin(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong( fmin(val, __longlong_as_double(assumed))) );
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
		} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void BellmanFord_split1cuda(int * finished, int V, int E, int *offsets, int *edge_dests,
		double *weights, int * preds, int * temp_preds, double * path_weights, double * temp_path_weights){
	//int my_vert = blockIdx.x;
	int my_vert = blockIdx.x *blockDim.x + threadIdx.x;

	if(my_vert < V) {
		int my_dist;
		int first_target_index, last_target_index, target_index, target;
		double new_dist;
		my_dist = path_weights[my_vert];

		//Find bounds of adjacency list
		first_target_index = offsets[my_vert];
		if(my_vert != V-1){
			last_target_index = offsets[my_vert+1];
		}
		else{
			last_target_index = E;
		}

		for(target_index = first_target_index; target_index < last_target_index; target_index++){
			target = edge_dests[target_index];
			new_dist = my_dist + weights[target_index];
			// need to change path_weights[target] and update predecessors[target]
			// Try to make this atomic
			if(new_dist < atomicMin(&temp_path_weights[target], new_dist) ){
				temp_preds[target] = my_vert;
				preds[target] = my_vert;
				*finished = 0;
				//temp_path_weights[target] = new_dist;
			}
		}
	}
}

__global__ void BellmanFord_split2cuda(int V, int E, int *offsets, int *edge_dests, double *weights, int * preds, int * temp_preds, double * path_weights, double * temp_path_weights){
	//int my_vert = blockIdx.x;
	int my_vert = blockIdx.x *blockDim.x + threadIdx.x;
	int first_target_index, last_target_index;
	int pred_vert;

	if(my_vert < V){
		pred_vert = temp_preds[my_vert];
		if(pred_vert >= 0 && pred_vert != my_vert){
			//Update predecessors
			preds[my_vert] = pred_vert;

			//Find bounds of adjacency list of pred_vert, want to find edge that leads to my_vert
			first_target_index = offsets[pred_vert];
			if(pred_vert != V-1){
				last_target_index = offsets[pred_vert+1];
			}
			else{
				last_target_index = E;
			}

			//Update path_weights
			for(int i=first_target_index; i < last_target_index; i++){
				if(edge_dests[i] == my_vert){
					temp_path_weights[my_vert] = path_weights[pred_vert] + weights[i];
					break;
				}
			}
		}
	}
}


double CSR_Graph::BellmanFordGPU_Split(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight){
	int num_blocks = (V + threads_per_block - 1) / threads_per_block;

	//Initialize predecessor tree
	predecessors.clear();
	path_weight.clear();
	double inf = std::numeric_limits<double>::infinity();
	predecessors.resize(V,-1);
	path_weight.resize(V,E*max_weight);
	predecessors[source_]=source_;
	path_weight[source_]=0;
	int finished;

	boost::timer::auto_cpu_timer t;

	//GPU pointers
	int *  d_offsets;
	int * d_edge_dests;
	double * d_weights;
	int * d_predecessors;
	double * d_path_weight;
	int * d_temp_predecessors;
	double * d_temp_path_weight;
	int * d_finished;
	double * temp;

	//Size of CSR graph
	int offsets_size = V*sizeof(int);
	int edge_dests_size = E*sizeof(int);
	int weights_size = E*sizeof(double);

	//Size of predecessor tree into
	int predecessors_size = V*sizeof(int);
	int temp_predecessors_size = V*sizeof(int);
	int path_weight_size = V*sizeof(double);

	//Allocate memory on device
	cudaMalloc((void **) & d_offsets, offsets_size);
	cudaMalloc((void **) & d_edge_dests, edge_dests_size);
	cudaMalloc((void **) & d_weights, weights_size);
	cudaMalloc((void **) & d_predecessors, predecessors_size);
	cudaMalloc((void **) & d_temp_predecessors, temp_predecessors_size);
	cudaMalloc((void **) & d_path_weight, path_weight_size);
	cudaMalloc((void **) & d_temp_path_weight, path_weight_size);
	cudaMalloc((void **) & d_finished, sizeof(int));

	std::cout<<"Transferring to GPU"<<std::endl;
	cudaMemcpy(d_offsets, (int *) &offsets[0], offsets_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_dests, (int *) &edge_dests[0], edge_dests_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, (double *) &weights[0], weights_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_predecessors, (int *) &predecessors[0], predecessors_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp_predecessors, (int *) &predecessors[0], temp_predecessors_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_path_weight, (double *) &path_weight[0], path_weight_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_temp_path_weight, (double *) &path_weight[0], path_weight_size, cudaMemcpyHostToDevice);


	std::cout<<"Running kernel with <<<" << num_blocks << ", " << threads_per_block << ">>>" <<std::endl;
	int iter=0;
	finished=0;
	boost::timer::cpu_timer timer;
//	for(int iter=0; iter<V; iter++){
	while(finished == 0 && iter < 200) {
		std::cout<<"Iter = "<<iter<<std::endl;
		finished=1;
		std::cout<<finished<<std::endl;

		cudaMemcpy(d_finished, &finished, sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		BellmanFord_split1cuda<<<num_blocks, threads_per_block>>>(d_finished, V, E, d_offsets,d_edge_dests,
				d_weights,d_predecessors,d_temp_predecessors,d_path_weight, d_temp_path_weight);
		cudaDeviceSynchronize();

		cudaMemcpy(&finished, d_finished,  sizeof(int), cudaMemcpyDeviceToHost);
//		BellmanFord_split2cuda<<<num_blocks, threads_per_block>>>(V, E, d_offsets,d_edge_dests,
//				d_weights,d_predecessors,d_temp_predecessors,d_path_weight, d_temp_path_weight);
//		cudaDeviceSynchronize();

		cudaMemcpy(d_path_weight, d_temp_path_weight, path_weight_size, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();

		cudaMemcpy((double *) &path_weight[0], d_path_weight, path_weight_size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		for(int i=0; i<V; i++) {
			std::cout<<"V: "<<i<<",\t PW: "<<path_weight[i]<<std::endl;
		}

		std::cout<<finished<<std::endl;

//		temp=d_path_weight;
//		d_path_weight = d_temp_path_weight;
//		d_temp_path_weight = temp;

		iter++;
	}
//	}

	timer.stop();

	//Copy results back to host
	//cudaMemcpy((int *) &offsets[0], d_offsets, offsets_size, cudaMemcpyDeviceToHost);
	//cudaMemcpy((int *) &edge_dests[0], d_edge_dests, edge_dests_size, cudaMemcpyDeviceToHost);
	//cudaMemcpy((double *) &weights[0], d_weights, weights_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((int *) &predecessors[0], d_predecessors, predecessors_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((double *) &path_weight[0], d_path_weight, path_weight_size, cudaMemcpyDeviceToHost);

	//cleanup
	cudaFree(d_offsets); cudaFree(d_edge_dests); cudaFree(d_weights);
	cudaFree(d_predecessors); cudaFree(d_path_weight); cudaFree(d_temp_predecessors);

	for(int i=0; i<V; i++){
		if(path_weight[i] == E*max_weight){
			path_weight[i] = inf;
		}
	}

	return (double) timer.elapsed().wall / 1000000000.0;
}

__global__ void BellmanFord_cuda(int V, int E, int *offsets, int *edge_dests, double *weights, int * preds, double * path_weights){
	//int my_vert = blockIdx.x;
	int my_vert = blockIdx.x *blockDim.x + threadIdx.x;
	//int my_vert = threadIdx.x;

	if(my_vert < V) {
		int source_vert;

		double my_dist = path_weights[my_vert];
		double trial_dist;

		source_vert=0;
		for(int i=0; i<E; i++){
			if(edge_dests[i] == my_vert){
				//we can keep track of what the source vertex could be, since the edge list is sorted by them
				while(source_vert != V-1  && offsets[source_vert+1] <= i){
					source_vert++;
				}
				trial_dist = weights[i] + path_weights[source_vert]; //Data race, possibly benign?
				if(trial_dist < my_dist){
					path_weights[my_vert] = trial_dist;
					preds[my_vert] = source_vert;
				}
			}
		}
	}
}

double CSR_Graph::BellmanFordGPU(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight){
	int num_blocks = (V + threads_per_block - 1) / threads_per_block;

	//Initialize predecessor tree
	predecessors.clear();
	path_weight.clear();
	double inf = std::numeric_limits<double>::infinity();
	predecessors.resize(V,-1);
	path_weight.resize(V,E*max_weight);
	predecessors[source_]=source_;
	path_weight[source_]=0;

	boost::timer::auto_cpu_timer t;

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

	std::cout<<"Running kernel with <<<" << num_blocks << ", " << threads_per_block << ">>>" <<std::endl;
	boost::timer::cpu_timer timer;
	for(int iter=0; iter<V; iter++){
		//std::cout<<iter<<std::endl;
		BellmanFord_cuda<<<num_blocks, threads_per_block>>>(V, E, d_offsets,d_edge_dests,d_weights,d_predecessors,d_path_weight);
		cudaDeviceSynchronize();
	}
	timer.stop();

	//Copy results back to host
	cudaMemcpy((int *) &offsets[0], d_offsets, offsets_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((int *) &edge_dests[0], d_edge_dests, edge_dests_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((double *) &weights[0], d_weights, weights_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((int *) &predecessors[0], d_predecessors, predecessors_size, cudaMemcpyDeviceToHost);
	cudaMemcpy((double *) &path_weight[0], d_path_weight, path_weight_size, cudaMemcpyDeviceToHost);

	//cleanup
	cudaFree(d_offsets); cudaFree(d_edge_dests); cudaFree(d_weights);
	cudaFree(d_predecessors); cudaFree(d_path_weight);

	return (double) timer.elapsed().wall / 1000000000.0;
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

#include "cuda_test_class.h"

__global__ void add(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

cuda_test_class::cuda_test_class(int N_){
	N=N_;
	a = (int *) malloc(N);
	b = (int *) malloc(N);
	c = (int *) malloc(N);

	for(int i=0; i<N; i++){
		a[i] = i;
		b[i] = i*i;
	}
}

bool cuda_test_class::check(){
	for(int i=0; i<N; i++){
		if(a[i] + b[i] != c[i]){
			std::cout<<"Error! Sum not correct for index "<<i<<std::endl;
			std::cout<<a[i]<<" + "<<b[i]<<" != "<<c[i]<<std::endl;
			return false;
		}
	}
	return true;
}

void cuda_test_class::add(){
	int *d_size;
	int *d_a, *d_b, *d_c;

	int size=N*sizeof(int);
	cudaMalloc((void **) & d_a, size);
	cudaMalloc((void **) & d_b, size);
	cudaMalloc((void **) & d_c, size);

	//Copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add<<<N,1>>>(d_a,d_b);

	//Copy results back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	//cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

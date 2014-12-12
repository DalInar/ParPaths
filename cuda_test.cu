
#include <iostream>
#include "cuda_test_class.h"

__global__ void mykernel(int *a, int * b, int * c){
	*c=*a+*b;
}

int main(void){
	int a,b,c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	//Allocate space for device
	cudaMalloc((void **) & d_a, size);
	cudaMalloc((void **) & d_b, size);
	cudaMalloc((void **) & d_c, size);

	a=2;
	b=7;

	//Copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	mykernel<<<1,1>>>(d_a, d_b, d_c);

	//Copy results back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	//cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	std::cout<<"CUDA answer = "<<c<<std::endl;
	std::cout<<"Should be = "<<a+b<<std::endl;

	int N=10;
	cuda_test_class c_test(N);
	std::cout<<"Created class"<<std::endl;
	std::cout<<"Now check"<<std::endl;
	std::cout<<"Passed? "<<c_test.check()<<std::endl;
	std::cout<<"GPU adding"<<std::endl;
	c_test.add();
	std::cout<<"Now check gpu"<<std::endl;
	std::cout<<"Passed? "<<c_test.check()<<std::endl;

	std::cout<<"Hello World!"<<std::endl;
	return 0;
}

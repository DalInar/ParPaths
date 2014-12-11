/*
 * cuda_test_class.h
 *
 *  Created on: Dec 11, 2014
 *      Author: pakij
 */

#ifndef CUDA_TEST_CLASS_H_
#define CUDA_TEST_CLASS_H_

#include <iostream>

class cuda_test_class{
public:
	cuda_test_class(int N_);

	void add();
	bool check();

private:
	int N;
	int *a;
	int *b;
	int *c;
};



#endif /* CUDA_TEST_CLASS_H_ */

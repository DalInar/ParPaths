/*
 * prioqueue.cpp
 *
 *  Created on: Dec 1, 2014
 *      Author: pakij
 */

#include "prioqueue.h"

prioqueue_vect::prioqueue_vect(std::vector <int> vals) {
	size=vals.size();
	remaining=size;
	Q=vals;
}

int prioqueue_vect::extract_min() {
	int min_val=Q[0];
	int min_index=0;

	for(int i=1; i<size; i++){
		if(Q[i] < min_val) {
			min_val=Q[i];
			min_index=i;
		}
	}
	remaining -= 1;
	Q[min_index] = std::numeric_limits<int>::max();
	return min_index;
}



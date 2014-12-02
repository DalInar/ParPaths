/*
 * prioqueue.h
 *
 *  Created on: Dec 1, 2014
 *      Author: pakij
 */

#ifndef PRIOQUEUE_H_
#define PRIOQUEUE_H_

#include <iostream>
#include <vector>
#include <limits>

class prioqueue_vect
{
public:
	prioqueue_vect(std::vector <int> vals);

	void decrease_key(int index, int val){Q[index]=val;}
	int extract_min();
	bool empty(){return remaining==0 ? true:false;}

private:
	std::vector <int> Q;
	int size;
	int remaining;
};



#endif /* PRIOQUEUE_H_ */

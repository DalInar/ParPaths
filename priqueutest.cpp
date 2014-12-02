/*
 * priqueutest.cpp
 *
 *  Created on: Dec 1, 2014
 *      Author: pakij
 */

#include <iostream>
#include <vector>
#include <queue>

int main() {
	std::priority_queue <int> priq;
	priq.push(5);
	priq.push(-1);
	priq.push(2);
	priq.push(5);
	priq.push(9);
	priq.push(17);

	while(!priq.empty()){

		std::cout<<priq.top()<<std::endl;
		priq.pop();

	}
	return 0;
}



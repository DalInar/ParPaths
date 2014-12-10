/*
 * CSR_test.cpp
 *
 *  Created on: Dec 4, 2014
 *      Author: pakij
 */

#include <iostream>
#include "CSR_Graph.h"

int main(){
	std::cout<<"Hello!"<<std::endl;
	CSR_Graph G = CSR_Graph(5,6,12.3);
	G.print_graph();

	std::cout<<"Finished!"<<std::endl;
	return 0;
}




/*
 * CSR_test.cpp
 *
 *  Created on: Dec 4, 2014
 *      Author: pakij
 */

#include <iostream>
#include <set>
#include "CSR_Graph.h"

int main(){
	std::cout<<"Hello!"<<std::endl;
	int V=6;
	CSR_Graph G = CSR_Graph(V,8,12.3);
	G.print_graph();

	int source=2;
	std::vector <int> predecessors;
	std::vector <double> path_weight;
	G.Dijkstra(source, predecessors, path_weight);

	std::cout<<"SSSP Valid: "<<G.validate(source, predecessors, path_weight)<<std::endl;
	path_weight[3]=2.3;
	std::cout<<"SSSP Valid: "<<G.validate(source, predecessors, path_weight)<<std::endl;
	for(int i=0; i<V; i++){
		std::cout<<"V: "<<i<<"\t Pred:"<<predecessors[i]<<"\t W:"<<path_weight[i]<<std::endl;
	}

	std::cout<<"Edge Weight between 1 and 3 = "<<G.get_edge_weight(1,3)<<std::endl;

	std::cout<<"Finished!"<<std::endl;
	return 0;
}




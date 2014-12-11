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
	int V=10;
	CSR_Graph G = CSR_Graph(V,15,12.3);
	G.print_graph();

	int source=2;
	std::vector <int> predecessors;
	std::vector <double> path_weight;
	std::vector <int> predecessors_BF;
	std::vector <double> path_weight_BF;
	G.Dijkstra(source, predecessors, path_weight);
	//G.BellmanFord(source, predecessors_BF, path_weight_BF);

	std::cout<<"Dijkstra SSSP Valid: "<<G.validate(source, predecessors, path_weight)<<std::endl;
	//std::cout<<"Bellman-Ford SSSP Valid: "<<G.validate(source, predecessors_BF, path_weight_BF)<<std::endl;
	std::cout<<"Does Dijkstra equal Bellman-Ford? "<< ((predecessors==predecessors_BF) && (path_weight== path_weight_BF)) << std::endl;

//	for(int i=0; i<V; i++){
//		std::cout<<"V: "<<i<<"\t Pred:"<<predecessors_BF[i]<<"\t W:"<<path_weight_BF[i]<<std::endl;
//	}

	std::string filename = "output.txt";
	G.print_graph_to_file(filename);
	CSR_Graph G_new = CSR_Graph(filename);
	G_new.print_graph();

	std::cout<<"Finished!"<<std::endl;
	return 0;
}




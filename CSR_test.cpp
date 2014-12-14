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
	G.BellmanFord(source, predecessors_BF, path_weight_BF);

	std::cout<<"Dijkstra SSSP Valid: "<<G.validate(source, predecessors, path_weight)<<std::endl;
	std::cout<<"Bellman-Ford SSSP Valid: "<<G.validate(source, predecessors_BF, path_weight_BF)<<std::endl;
	std::cout<<"Does Dijkstra equal Bellman-Ford? "<< ((predecessors==predecessors_BF) && (path_weight== path_weight_BF)) << std::endl;

//	for(int i=0; i<V; i++){
//		std::cout<<"V: "<<i<<"\t Pred:"<<predecessors_BF[i]<<"\t W:"<<path_weight_BF[i]<<std::endl;
//	}

	std::cout<<std::endl<<"New tests:"<<std::endl;
	std::vector <int> predecessors_BF_gpu;
	std::vector <double> path_weight_BF_gpu;
	std::string filename = "output.txt";
	V=10000;
	CSR_Graph G_gpu = CSR_Graph(V,100000,12.3);
	//G_gpu.print_graph();
	G_gpu.print_graph_to_file(filename);
	std::cout<<"Bellman-Ford time: "<<G_gpu.BellmanFord(source, predecessors_BF, path_weight_BF)<<std::endl;
	std::cout<<"Dijkstra time: "<<G_gpu.Dijkstra(source, predecessors, path_weight)<<std::endl;
	std::cout<<"Bellman-Ford GPU time: "<<G_gpu.BellmanFordGPU(source, predecessors_BF_gpu, path_weight_BF_gpu)<<std::endl;

	std::cout<<"Are they equal? "<< ((predecessors==predecessors_BF) && (path_weight== path_weight_BF)) << std::endl;
	for(int i=0; i < 5; i++){
		std::cout<<"V: "<<i<<std::endl;
		std::cout<<predecessors[i]<<"\t"<<predecessors_BF[i]<<std::endl;
		std::cout<<path_weight[i]<<"\t"<<path_weight_BF[i]<<std::endl;
	}

	std::cout<<"Finished!"<<std::endl;
	return 0;
}




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
	int E;
//	CSR_Graph G = CSR_Graph(V,15,12.3);
//	G.print_graph();
//
	int source=2;
	std::vector <int> predecessors;
	std::vector <double> path_weight;
	std::vector <int> predecessors_BF;
	std::vector <double> path_weight_BF;
	std::vector <int> predecessors_BF_gpu;
	std::vector <double> path_weight_BF_gpu;
//	G.Dijkstra(source, predecessors, path_weight);
//	G.BellmanFord(source, predecessors_BF, path_weight_BF);
//	G.BellmanFordGPU(source, predecessors_BF_gpu, path_weight_BF_gpu);
//
//	std::cout<<"Dijkstra SSSP Valid: "<<G.validate(source, predecessors, path_weight)<<std::endl;
//	std::cout<<"Bellman-Ford SSSP Valid: "<<G.validate(source, predecessors_BF, path_weight_BF)<<std::endl;
//	std::cout<<"Bellman-Ford GPU SSSP Valid: "<<G.validate(source, predecessors_BF_gpu, path_weight_BF_gpu)<<std::endl;
//	std::cout<<"Does Dijkstra equal Bellman-Ford? "<< ((predecessors==predecessors_BF) && (path_weight== path_weight_BF)) << std::endl;
//	std::cout<<"Does Dijkstra equal Bellman-Ford GPU? "<< (predecessors==predecessors_BF_gpu) << std::endl;
//	for(int i=0; i<V; i++){
//		std::cout<<"V: "<<i<<"\t Pred:"<<predecessors_BF[i]<<"\t W:"<<path_weight_BF[i]<<std::endl;
//	}

	std::cout<<std::endl<<"New tests:"<<std::endl;

	std::string filename = "output.txt";
	V=10000;
	CSR_Graph G_gpu = CSR_Graph(V,1000000,12.3);
//	G_gpu.print_graph();
	G_gpu.print_graph_to_file(filename);

	std::cout<<"Bellman-Ford time: "<<G_gpu.BellmanFord(source, predecessors_BF, path_weight_BF)<<std::endl;
	std::cout<<"Dijkstra time: "<<G_gpu.Dijkstra(source, predecessors, path_weight)<<std::endl;
	std::cout<<"Size = "<<predecessors.size()<<std::endl;

	std::cout<<"Bellman-Ford GPU time: "<<G_gpu.BellmanFordGPU_Split(source, predecessors_BF_gpu, path_weight_BF_gpu)<<std::endl;
	std::cout<<"Dijkstra SSSP Valid: "<<G_gpu.validate(source, predecessors, path_weight)<<std::endl;

	std::cout<<"Are preds equal? "<< (predecessors==predecessors_BF_gpu)<< std::endl;
	std::cout<<"Are PWs equal? "<< (predecessors==predecessors_BF_gpu)<< std::endl;
	for(int i=0; i < 5; i++){
		std::cout<<"V: "<<i<<std::endl;
		std::cout<<"Pred: "<< predecessors[i]<<"\t"<<predecessors_BF_gpu[i]<<std::endl;
		std::cout<<"PW: "<<path_weight[i]<<"\t"<<path_weight_BF_gpu[i]<<std::endl;
	}

	std::ofstream GPU_time;
	GPU_time.open("BF_GPU_Time.txt");
	int threads_per_block = 1;
	while(threads_per_block < (V+31) && threads_per_block <= 1024){
		G_gpu.set_threads_per_block(threads_per_block);
		GPU_time<<threads_per_block<<"\t"<<G_gpu.BellmanFordGPU_Split(source, predecessors_BF_gpu, path_weight_BF_gpu)<<std::endl;
		if(threads_per_block ==1){
			threads_per_block = 32;
		}
		else{
			threads_per_block += 32;
		}
	}
	GPU_time.close();

	std::cout<<"Finished!"<<std::endl;


	std::ofstream serial_time;
	serial_time.open("serial_timing.txt");

	int serial_source = 0;
	std::vector <int> serial_preds;
	std::vector <double> serial_path_weights;

	double scale = 0.2;
	serial_time<<"# scale = "<<scale<<std::endl;
	serial_time<<"#V \t E \t BF \t Dij"<<std::endl;
	for(V=100; V<=10100; V=V+500){
		E=scale*V*V;
		CSR_Graph G_temp = CSR_Graph(V,E,100);

		serial_time<<V<<"\t"<<E<<"\t";

		serial_time << G_temp.BellmanFord(serial_source, serial_preds, serial_path_weights) <<"\t";
		serial_time << G_temp.Dijkstra(serial_source, serial_preds, serial_path_weights) << std::endl;
	}


	return 0;
}




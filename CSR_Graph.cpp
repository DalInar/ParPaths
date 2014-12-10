/*
 * CSR_Graph.cpp
 *
 *  Created on: Dec 4, 2014
 *      Author: pakij
 */

#include "CSR_Graph.h"

CSR_Graph::CSR_Graph(int V_, int E_, double max_weight_):V(V_),E(E_),max_weight(max_weight_) {
	srand (time(NULL));

	//Temporary adjacency matrix to assist in creating random graph
	std::vector< std::vector<double> > Adj(V, std::vector<double>(V));
	for(int i=0; i<V; i++){
		for(int j=0; j<V; j++){
			Adj[i][j]=-1.0;
		}
	}

	//Generate a random graph by randomly picking edges
	int E_temp=0;
	int s,d;
	while(E_temp < E){
		s=rand()%V;
		d=rand()%V;
		if(s != d && Adj[s][d] < 0){
			Adj[s][d] = (rand()/(double)((double)RAND_MAX + 1)) * max_weight;
			E_temp += 1;
		}
	}

	E_temp=0;
	for(int s=0; s<V; s++) {
		offsets.push_back(E_temp);
		for(int d=0; d<V; d++) {
			if(Adj[s][d] >= 0) {
				E_temp += 1;
				edge_dests.push_back(d);
				weights.push_back(Adj[s][d]);
			}
		}
	}
}

void CSR_Graph::Dijkstra(int source, std::vector <int> &predecessors, std::vector <double> &path_weight) {
	predecessors.clear();
	path_weight.clear();
	predecessors.resize(V,-1);
	path_weight.resize(V,-1.0);
}

void CSR_Graph::print_graph() {
	std::cout<<"Print out of Graph Adjacency list"<<std::endl;
	std::cout<<"V = "<<V<<", E = "<<E<<std::endl;
	std::cout<<"Source: Dest(Weight), ..."<<std::endl;

	for(int s=0; s<V-1; s++) {
		std::cout<<s<<": ";
		for(int i=offsets[s]; i<offsets[s+1]; i++) {
			std::cout<<edge_dests[i]<<" ("<<weights[i]<<"), ";
		}
		std::cout<<std::endl;
	}
	std::cout<<V-1<<": ";
	for(int i=offsets[V-1]; i<E; i++) {
		std::cout<<edge_dests[i]<<" ("<<weights[i]<<"), ";
	}
	std::cout<<std::endl;
}



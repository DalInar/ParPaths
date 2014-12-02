/*
 * Dijkstra.cpp
 *
 *  Created on: Dec 1, 2014
 *      Author: pakij
 */

#include <iostream>
#include <queue>
#include "Graph.h"

int main(){
//	Vertex a_vert(1);
//	Vertex b_vert(2,4,&a_vert);
//	//ert.set_d(5);
//
//	a_vert.print();
//	b_vert.print();
//	std::cout<<(b_vert.get_pred())->get_label()<<std::endl;

	int num_vert=6;

	int sourceints[] = {0,4,0,2,1,4,0};
	std::vector<int> sources (sourceints, sourceints + sizeof(sourceints) / sizeof(int) );

	int destints[] = {3,1,1,0,4,5,5};
	std::vector<int> dests (destints, destints + sizeof(destints) / sizeof(int) );

	int weightints[] = {16,2,77,29,8,35,344};
	std::vector<int> weights (weightints, weightints + sizeof(weightints) / sizeof(int) );

	Graph g(num_vert, sources, dests, weights);
	g.print();

	g.Dijkstra(0);
	g.print();

	g.print_path(5);
	g.print_path(4);
	return 0;
}



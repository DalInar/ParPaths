/*
 * CSR_Graph.h
 *
 *  Created on: Dec 4, 2014
 *      Author: pakij
 */

#ifndef CSR_GRAPH_H_
#define CSR_GRAPH_H_

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include <limits>
#include <boost/timer/timer.hpp>
#include <set>
#include <fstream>

struct Vertex{
	Vertex(int idx_, double dist_) : idx(idx_), dist(dist_) { }
	int idx;
	double dist;
};

bool operator<(const Vertex &a, const Vertex &b);


class CSR_Graph
{
public:
	CSR_Graph(): V(0),E(0),max_weight(0), threads_per_block(1) {}
	CSR_Graph(int V_, int E_);	//Generate random graph with all weights=1
	CSR_Graph(int V_, int E_, double max_weight_); //Generate random graph with random weights, bounded by max_weight
	CSR_Graph(std::string filename);

	void print_graph();
	bool print_graph_to_file(std::string filename);

	double Dijkstra(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight);
	double BellmanFord(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight);

	double BellmanFordGPU(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight);

	double BellmanFordAtomicGPU(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight);

	bool validate(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight);
	double get_edge_weight(int source_, int dest_);

	void set_threads_per_block(int threads_per_block_) {threads_per_block = threads_per_block_;}

	bool test_cuda();


private:
	int V;
	int E;
	double max_weight;
	std::vector <int> offsets;
	std::vector <int> edge_dests;
	std::vector <double> weights;

	int threads_per_block;
};

#endif /* CSR_GRAPH_H_ */

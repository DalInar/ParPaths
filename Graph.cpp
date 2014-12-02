/*
 * Graph.cpp
 *
 *  Created on: Dec 1, 2014
 *      Author: pakij
 */

#include "Graph.h"

void Vertex::print() {
	std::cout<<"Vertex "<<label<<std::endl;
	std::cout<<"d = "<<d<<std::endl;
	if(pred != 0) {
		std::cout<<"pred = "<<pred->get_label()<<std::endl;
	}
	else {
		std::cout<<"No predecessor"<<std::endl;
	}
}

Graph::Graph(int size, std::vector<int> sources, std::vector<int> dests, std::vector<int> weights) {
	V=size;
	E=sources.size();

	for(int i=0; i<V; i++){
		vertices.push_back(Vertex (i));
		adjacency.push_back(std::list<Edge> (0));
	}

	for(int i=0; i<E; i++) {
		int source = sources[i];
		Edge e(weights[i], sources[i], dests[i]);
		adjacency[source].push_back(Edge (weights[i],sources[i],dests[i], &vertices[source], &vertices[dests[i]]));
	}
	source_vertex=0;
}

Graph::Graph(int size):E(0) {
	V=size;
	for(int i=0; i<V; i++){
		vertices.push_back(Vertex (i));
	}
	source_vertex=0;
}

void Graph::initialize(int s){
	for(int i=0; i<V; i++) {
		Vertex* v = &vertices[i];
		if(s == v->get_label()){
			v->set_pred(0);
			v->set_d(0);
		}
		else{
			v->set_pred(0);
			v->set_d(std::numeric_limits<int>::max());
		}
	}
}

void Graph::relax(int source, prioqueue_vect & Q){
	int d = vertices[source].get_d();
	std::cout<<"Relaxing "<<source<<std::endl;
	//std::cout<<vertices[source].get_d()<<std::endl;
	for(std::list<Edge>::iterator iterator = adjacency[source].begin(), end = adjacency[source].end(); iterator != end; ++iterator) {
		int dest = iterator->get_dest_label();
		int w = iterator->get_weight();
		if(d+w >= 0 && vertices[dest].get_d() > d+w){
			//std::cout<<vertices[dest].get_d()<<std::endl;
			//std::cout<<d+w<<std::endl;
			vertices[dest].set_d(d+w);
			Q.decrease_key(dest,d+w);
			vertices[dest].set_pred(&vertices[source]);
		}
	}
}

void Graph::Dijkstra(int s) {
	initialize(s);
	source_vertex=s;

	std::vector <int> init_dist(V,std::numeric_limits<int>::max());
	init_dist[s] = 0;
	prioqueue_vect Q(init_dist);

	while(!Q.empty()){
		relax(Q.extract_min(), Q);
	}
}

void Graph::print() {
	for(int i=0; i<V; i++) {
		std::cout<<"V"<<vertices[i].get_label()<<": ";
		for(std::list<Edge>::iterator iterator = adjacency[i].begin(), end = adjacency[i].end(); iterator != end; ++iterator) {
		    std::cout << iterator->get_dest_label() <<"("<<iterator->get_weight()<<"), ";
		}
		std::cout<<"d = "<<vertices[i].get_d()<<", ";
		if(vertices[i].get_pred() != 0){
			std::cout<<"Pred = "<<vertices[i].get_pred()->get_label();
		}
		else{
			std::cout<<"No Pred";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
}

void Graph::print_path(int dest) {
	if(vertices[dest].get_pred() == 0) {
		std::cout<<"There is no path from vertex "<<source_vertex<<" to vertex "<<dest<<std::endl;
	}
	else{
		std::cout<<"The shortest path from vertex "<<source_vertex<<" to vertex "<<dest<<" has weight "<<vertices[dest].get_d()<<std::endl;
		Vertex* v = &vertices[dest];
		while(v->get_pred() != 0){
			std::cout<<v->get_label()<<" <- ";
			v= v->get_pred();
		}
		std::cout<<v->get_label()<<std::endl;
	}
}

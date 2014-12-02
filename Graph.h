/*
 * Graph.h
 *
 *  Created on: Dec 1, 2014
 *      Author: pakij
 */

#include <iostream>
#include <limits>
#include <list>
#include <vector>
#include <queue>
#include "prioqueue.h"

class Vertex
{
public:
	Vertex(int label_):
		label(label_),
		d(std::numeric_limits<int>::max()),
		pred(0){}
	Vertex(int label_, int d_, Vertex* pred_):
		label(label_),
		d(d_),
		pred(pred_){}

	void set_label(int label_){label=label_;}
	void set_d(int d_){d=d_;}
	void set_pred(Vertex* pred_){pred=pred_;}

	int get_label(){return label;}
	int get_d(){return d;}
	Vertex* get_pred(){return pred;}

	void print();

private:
	int label;
	int d;
	Vertex* pred;
};

//class CompareVertex {
//    public:
//    bool operator()(Vertex* v1, Vertex* v2) // v2 has higher priority if v2 has a shorter path than v1
//    {
//       if (v2->get_d() < v1->get_d()) return true;
//       return false;
//    }
//};

class Edge {
public:
	Edge():
		weight(-1),
		source_label(-1),
		dest_label(-1),
		source_vert(0),
		dest_vert(0){}
	Edge(int weight_, int source_label_, int dest_label_):
		weight(weight_),
		source_label(source_label_),
		dest_label(dest_label_),
		source_vert(0),
		dest_vert(0){}

	Edge(int weight_, int source_label_, int dest_label_, Vertex* source_vert_, Vertex* dest_vert_):
		weight(weight_),
		source_label(source_label_),
		dest_label(dest_label_),
		source_vert(source_vert_),
		dest_vert(dest_vert_){}

	void set_weight(int weight_){weight=weight_;}
	void set_source_label(int source_label_){source_label=source_label_;}
	void set_dest_label(int dest_label_){dest_label=dest_label_;}
	void set_source_vert(Vertex* source_vert_){source_vert=source_vert_;}
	void set_dest_vert(Vertex* dest_vert_){dest_vert=dest_vert_;}

	int get_weight(){return weight;}
	int get_source_label(){return source_label;}
	int get_dest_label(){return dest_label;}
	Vertex* get_source_vert(){return source_vert;}
	Vertex* get_dest_vert(){return dest_vert;}

private:
	int weight;
	int source_label;
	int dest_label;
	Vertex * source_vert;
	Vertex * dest_vert;
};

class Graph
{
public:
	Graph():V(0),E(0),source_vertex(0){}
	Graph(int size);
	Graph(int size, std::vector<int> sources, std::vector<int> dests, std::vector<int> weights);

	int num_vert(){return V;}
	int num_edge(){return E;}

	void initialize(int s);
	void relax(int source, prioqueue_vect & Q);
	void Dijkstra(int s);
	void print();
	void print_path(int dest);

private:
	int V;	//Number of vertices
	int E;	//Number of edges
	std::vector< Vertex > vertices;
	std::vector< std::list<Edge> > adjacency;
	int source_vertex;

};




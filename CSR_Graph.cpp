/*
 * CSR_Graph.cpp
 *
 *  Created on: Dec 4, 2014
 *      Author: pakij
 */

#include "CSR_Graph.h"

bool operator<(const Vertex &a, const Vertex &b){
	return (a.dist < b.dist) && (a.idx != b.idx);
}

CSR_Graph::CSR_Graph(std::string filename):V(0),E(0),max_weight(-1){
	std::ifstream input;
	input.open(filename.c_str());
	if(!input.is_open()){
		std::cout<<"Error, file "<<filename<<" does not exist!"<<std::endl;
		exit(1);
	}

	input >> V;
	input >> E;
	input >> max_weight;

	std::string data;
	std::getline(input, data);

	offsets.resize(V,0);
	edge_dests.clear();
	weights.clear();

	int u, v, num_edges, insert_index;
	double weight;

	std::cout<<V<<std::endl;
	num_edges=0;
	while(num_edges != E){
		input >> u;
		input >> v;
		input >> weight;
		std::cout<<u<<" "<<v<<" "<<weight<<std::endl;
		num_edges+=1;

		for(int i=u+1; i<V; i++){
			offsets[i] += 1;
		}

		if(u != V-1){
			insert_index = offsets[u];
			edge_dests.insert(edge_dests.begin() + insert_index, v);
			weights.insert(weights.begin() + insert_index, weight);
		}
		else{
			insert_index = num_edges;
			edge_dests.push_back(v);
			weights.push_back(weight);
		}
	}
	input.close();
}

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

void CSR_Graph::BellmanFord(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight){
	std::clock_t start;
	std::cout<<"Finding shortest paths from V: "<<source_<<" with Bellman-Ford"<<std::endl;
	predecessors.clear();
	path_weight.clear();
	double inf = std::numeric_limits<double>::infinity();
	predecessors.resize(V,-1);
	path_weight.resize(V,inf);

	predecessors[source_]=source_;
	path_weight[source_]=0;

	int first_target_index, last_target_index, v;
	double edge_weight,u_dist;
	start=std::clock();
	boost::timer::auto_cpu_timer t;
	for(int i=0; i<V; i++){
		for(int u=0; u<V; u++){
			//Cycle through adjacency list of vertex u
			u_dist = path_weight[u];
			first_target_index=offsets[u];
			if(u != V-1){
				last_target_index=offsets[u+1];
			}
			else{
				last_target_index=E;
			}
			//Scan through adjacency list of c_vert
			for(int target_index=first_target_index; target_index<last_target_index; target_index++){
				v = edge_dests[target_index];
				edge_weight = weights[target_index];
				//Check if edge should be relaxed
				if(path_weight[v] > u_dist + edge_weight){
					predecessors[v] = u;
					path_weight[v] = u_dist + edge_weight;
				}
			}
		}
	}
	std::cout << "Bellman-Ford Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
}

double CSR_Graph::Dijkstra(int source, std::vector <int> &predecessors, std::vector <double> &path_weight) {
	std::cout<<"Finding shortest paths from V: "<<source<<" with Dijkstra"<<std::endl;
	predecessors.clear();
	path_weight.clear();
	double inf = std::numeric_limits<double>::infinity();
	predecessors.resize(V,-1);
	path_weight.resize(V,inf);

	//Keep track of where each vertex is in Q
	std::vector<std::multiset<Vertex>::iterator> v_iters(V);

	boost::timer::auto_cpu_timer t;
	boost::timer::cpu_timer timer;

	std::multiset<Vertex> Q;
	Q.insert(Vertex(source, 0));

	for(int i=0; i<V; i++){
		if(i!=source){
			v_iters[i] = Q.insert(Vertex(i,inf));
		}
	}

	std::multiset<Vertex>::iterator Viter=Q.begin();
//	std::cout<<"Initial Contents of Q"<<std::endl;
//	while(Viter != Q.end()){
//		std::cout<<(*Viter).idx<<"\t"<<(*Viter).dist<<std::endl;
//		Viter++;
//	}
//	std::cout<<std::endl;

	int c_vert, first_target_index, last_target_index;
	int d_vert;
	double edge_weight;;
	double c_dist;
	predecessors[source]=source;
	path_weight[source]=0;

	while(!Q.empty()){
		//Get next vertex with shortest dist in Q, remove from Q
		Viter=Q.begin();
		c_vert=(*Viter).idx;
		c_dist=(*Viter).dist;
		Q.erase(Viter);

		//Find out where c_vert's adjacency list starts and ends in edge_dests and weights
		first_target_index=offsets[c_vert];
		if(c_vert != V-1){
			last_target_index=offsets[c_vert+1];
		}
		else{
			last_target_index=E;
		}
		//Scan through adjacency list of c_vert
		for(int target_index=first_target_index; target_index<last_target_index; target_index++){
			d_vert = edge_dests[target_index];
			edge_weight = weights[target_index];
			//Check if edge should be relaxed
			if(path_weight[d_vert] > c_dist + edge_weight){
				Q.erase(v_iters[d_vert]);
				v_iters[d_vert] = Q.insert(Vertex(d_vert, c_dist+edge_weight));
				predecessors[d_vert] = c_vert;
				path_weight[d_vert] = c_dist+edge_weight;
			}
		}

	}
	timer.stop();
	timer.elapsed().wall;
	const boost::timer::nanosecond_type oneSecond(1000000000LL);
	return (double) timer.elapsed().wall / 1000000000.0;
}

double CSR_Graph::get_edge_weight(int source_, int dest_){
	//Find out where c_vert's adjacency list starts and ends in edge_dests and weights
	int first_target_index, last_target_index;
	first_target_index=offsets[source_];
	if(source_ != V-1){
		last_target_index=offsets[source_+1];
	}
	else{
		last_target_index=E;
	}
	//Scan through adjacency list of c_vert
	for(int target_index=first_target_index; target_index<last_target_index; target_index++){
		if(edge_dests[target_index]==dest_){
			return weights[target_index];
		}
	}
	return -1.0;
}

bool CSR_Graph::validate(int source_, std::vector <int> &predecessors, std::vector <double> &path_weight){

	if(path_weight[source_] != 0){
		std::cout<<"Error, distance from s to s is "<<path_weight[source_]<<std::endl;
		return false;
	}

	for(int v=0; v<V; v++){
		int p=predecessors[v];
		if(p<0){
			if(path_weight[v] != std::numeric_limits<double>::infinity()){
				std::cout<<"Error, vertex "<<v<<" unconnected to source "<<source_<<", but path weight = "<<path_weight[v]<<std::endl;
				return false;
			}
			continue;
		}
		if(v==source_){
			continue;
		}
		std::set <int> preds;

		//Check that the tree is a tree
		while((p>=0) && (p!=source_)){
			if(preds.insert(p).second == false){
				std::cout<<"Error, predecessor "<<p<<" seen twice from "<<v<<std::endl;
				return false;
			}
			p=predecessors[p];
		}

		p=predecessors[v];
		double edge_weight=get_edge_weight(p,v);

		//Check that d(s,v)=d(s,p(v))+w(p(v),v)
		if(edge_weight < 0 || path_weight[v] != (path_weight[predecessors[v]] + edge_weight)){
			std::cout<<"Error, paths weights from "<<source_<<" to "<<v<<" are incorrect"<<std::endl;
			std::cout<<edge_weight<<"\t"<<path_weight[v]<<"\t"<<path_weight[predecessors[v]] + edge_weight<<std::endl;
			return false;
		}

		//Check that edges cannot be relaxed
		int first_target_index, last_target_index;
		first_target_index=offsets[source_];
		if(source_ != V-1){
			last_target_index=offsets[source_+1];
		}
		else{
			last_target_index=E;
		}
		//Scan through adjacency list of c_vert
		for(int target_index=first_target_index; target_index<last_target_index; target_index++){
			int dest = edge_dests[target_index];
			edge_weight = weights[target_index];
			if(path_weight[dest] > path_weight[v] + edge_weight){
				std::cout<<"Error, edge from "<<v<<" to "<<dest<<" can be relaxed"<<std::endl;
				return false;
			}
		}
	}

	return true;
}

void CSR_Graph::print_graph() {
	std::cout<<"Print out of Graph Adjacency list"<<std::endl;
	std::cout<<"V = "<<V<<", E = "<<E<<", max_weight = "<<max_weight<<std::endl;
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

bool CSR_Graph::print_graph_to_file(std::string filename){
	std::ofstream output;
	output.open(filename.c_str());
	if(!output.is_open()){
		return false;
	}

	output<<V<<"\t"<<E<<"\t"<<max_weight<<std::endl;
	//output<<"Source \t Dest \t Weight  ..."<<std::endl;

	int first_target_index, last_target_index, v;
	double edge_weight;
	for(int u=0; u<V; u++){
		first_target_index=offsets[u];
		if(u != V-1){
			last_target_index=offsets[u+1];
		}
		else{
			last_target_index=E;
		}
		//Scan through adjacency list of c_vert
		for(int target_index=first_target_index; target_index<last_target_index; target_index++){
			v = edge_dests[target_index];
			edge_weight = weights[target_index];
			output << u << " " << v << " " << edge_weight<<std::endl;
		}
	}

//	for(int s=0; s<V-1; s++) {
//		output<<s<<"\t";
//		for(int i=offsets[s]; i<offsets[s+1]; i++) {
//			output<<edge_dests[i]<<"\t"<<weights[i]<<"\t";
//		}
//		output<<std::endl;
//	}
//	output<<V-1<<"\t";
//	for(int i=offsets[V-1]; i<E; i++) {
//		output<<edge_dests[i]<<"\t"<<weights[i]<<"\t";
//	}
//	output<<std::endl;

	output.close();
	return true;
}



#include <iostream>
#include <sstream>
#include <set>
#include "CSR_Graph.h"

int main(){
	std::vector <int> scales;
	std::vector <int> num_vert;
//	num_vert.push_back(10);
//	num_vert.push_back(100);
//	num_vert.push_back(1000);
	num_vert.push_back(10000);
	num_vert.push_back(100000);
	num_vert.push_back(1000000);

	std::cout<<"Test graph"<<std::endl;
	CSR_Graph gr(10,20,34.4);
	std::cout<<"End test graph"<<std::endl;


	scales.push_back(1);
	scales.push_back(5);
	scales.push_back(10);
	scales.push_back(50);
	scales.push_back(100);
//	scales.push_back(1000);

	int serial_source = 0;
	std::vector <int> serial_preds;
	std::vector <double> serial_path_weights;

	int scale;
	int V,E;
	for(int i=0; i<scales.size(); i++){
		scale = scales[i];
		std::ostringstream fileNameStream;
		fileNameStream << "CSR_timing_degree_"<< scale << ".txt";
		std::string fileName = fileNameStream.str();

		std::ofstream output;
		output.open(fileName.c_str());

		output<<"#scale% = "<<scale<<std::endl;
		output<<"#V \t E \t BF \t Dij \t BF GPU \t BF GPU SPLIT"<<std::endl;

		std::cout<<scale<<std::endl;
		for(int j=0; j<num_vert.size(); j++){
			V=num_vert[j];
			//E=(scale/100.0)*V*V;
			E=scale*V;
			CSR_Graph G_temp = CSR_Graph(V,E,100);
			G_temp.set_threads_per_block(64);

			std::cout<<V<<"\t"<<E<<std::endl;

			output<<V<<"\t"<<E<<"\t";

		//	output << G_temp.BellmanFord(serial_source, serial_preds, serial_path_weights) <<"\t";
			output << 0.0 << "\t";
			output << G_temp.Dijkstra(serial_source, serial_preds, serial_path_weights) << "\t";
		//	output << G_temp.BellmanFordGPU(serial_source, serial_preds, serial_path_weights) <<"\t";
			output << 0.0 << "\t";
			output << G_temp.BellmanFordGPU_Split(serial_source, serial_preds, serial_path_weights) <<std::endl;
		}

		output.close();
	}
}

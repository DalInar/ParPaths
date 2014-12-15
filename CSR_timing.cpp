#include <iostream>
#include <sstream>
#include <set>
#include "CSR_Graph.h"

int main(){
	std::vector <int> scales;

	scales.push_back(1);
	scales.push_back(10);
	scales.push_back(30);
	scales.push_back(60);
	scales.push_back(80);

	int serial_source = 0;
	std::vector <int> serial_preds;
	std::vector <double> serial_path_weights;

	int scale;
	int V,E;
	for(int i=0; i<scales.size(); i++){
		scale = scales[i];
		std::ostringstream fileNameStream("CSR_timing_scale_");
		fileNameStream << scale << ".txt";
		std::string fileName = fileNameStream.str();

		std::ofstream output;
		output.open(fileName.c_str());

		output<<"# 100 * scale% = "<<scale<<std::endl;
		output<<"#V \t E \t BF \t Dij \t BF GPU \t BF GPU SPLIT"<<std::endl;

		std::cout<<scale<<std::endl;
		for(V=100; V<=10100; V=V+1000){
			E=scale*V*V;
			CSR_Graph G_temp = CSR_Graph(V,E,100);
			G_temp.set_threads_per_block(64);

			std::cout<<V<<"\t"<<E<<std::endl;

			output<<V<<"\t"<<E<<"\t";

			output << G_temp.BellmanFord(serial_source, serial_preds, serial_path_weights) <<"\t";
			output << G_temp.Dijkstra(serial_source, serial_preds, serial_path_weights) << "\t";
			output << G_temp.BellmanFordGPU(serial_source, serial_preds, serial_path_weights) <<"\t";
			output << G_temp.BellmanFordGPU_Split(serial_source, serial_preds, serial_path_weights) <<std::endl;
		}

		output.close();
	}
}

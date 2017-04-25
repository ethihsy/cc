#include <iostream>
#include <random>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fstream>
#include <omp.h>
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"
#include "f.hpp"

using namespace std;

int main(){

	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	binarize_dataset(dataset);
	//srand(time(NULL));
	//check_data(dataset.training_images, 11);

	vector<vector<double>> x(batch, vector<double>(v_d+1));
	vector<Layer> nn;
	nn.push_back(Layer(batch, h_d));				//	h1
	nn.push_back(Layer(batch, n_s*(h_d+1)));	//	sh1
	nn.push_back(Layer(batch, n_s*h_d));		//	sh1y
	nn.push_back(Layer(batch, h_d));				//	h2
	nn.push_back(Layer(batch, n_s*(h_d+1)));	//	sh2
	nn.push_back(Layer(batch, n_s*v_d));		//	sh2y
	nn.push_back(Layer(batch, v_d));				//	y
	vector<vector<double>> loss(batch, vector<double>(v_d));

	vector<Weights> ww;
	ww.push_back(Weights(h_d, v_d+1));			//	wxh1
	ww.back().init();
	ww.push_back(Weights(h_d, h_d+1));			//	wh1h2
	ww.back().init();
	ww.push_back(Weights(v_d, h_d+1));			//	wh2y
	ww.back().init();

	int max_itr = dataset.training_images.size()/batch;
	double gg[max_itr], ll[max_itr];
	ofstream fs("train_loss.txt");

//	------------------------------------------------------------------------------	

	for (int epoch=0; epoch<max_epoch; epoch++){

//		shuffle(begin(dataset.training_images), end(dataset.training_images), default_random_engine(time(NULL)));
		memset(ll, 0, sizeof(ll));
		memset(gg, 0, sizeof(gg));

		for (int itr=0; itr<dataset.training_images.size(); itr+=batch){

			for (int i=0; i<nn.size(); i++)
				nn[i].reset();
			nn.back().seed();
			for (int i=0; i<ww.size(); i++)
				ww[i].reset();

			for (int i=0; i<batch; i++){
				x[i][v_d] = 1;
				for (int j=0; j<v_d; j++){
					x[i][j] = (double)dataset.training_images[i+itr][j];
					loss[i][j] = (double)dataset.training_images[i+itr][j+v_d];
				}
			}

			feed_forward(x, ww[0]._v, nn[0]._v);
			for (int i=0; i<5; i+=3){
				sample(nn[i]._v, nn[i+1]._v);
				feed_forward(nn[i+1]._v, ww[(i+3)/3]._v, nn[i+3]._v, nn[i+2]._v);
			}
			loss_function(nn.back()._v, nn.back()._a, loss);

			for (int i=nn.size()-1; i>2; i-=3){
				back_prop(nn[i]._a, nn[i]._v, nn[i-1]._a, nn[i-1]._v, ww[i/3]._a, ww[i/3]._v, nn[i-2]._a, nn[i-2]._v);
				a_sample(nn[i-2]._a, nn[i-2]._v, nn[i-3]._a, nn[i-3]._v, nn[i]._v);
			}
			back_prop_w(nn[0]._a, nn[0]._v, ww[0]._a, x);
			
	//		check_gd(2,2,0, nn, ww, x, dataset.training_images,loss);
			
			for (int i=0; i<ww.size(); i++)
				optimizer(ww[i]._v, ww[i]._a, ww[i]._m);

			for (int i=0; i<batch; i++)
				for (int j=0; j<v_d; j++)
					ll[itr/batch] += loss[i][j];
			ll[itr/batch] /= batch;

			for (int i=0; i<ww.size(); i++)
				gg[itr/batch] += ww[i].norm();
			gg[itr/batch] /= ww.size();

		}
		for (int i=0; i<max_itr; i++)
			fs << i+epoch*max_itr << "\t" << ll[i] << "\t" << gg[i] << endl;
		cout << "epoch: " << epoch << " is done" << endl;
		
	}
	fs.close();
	for (int i=0; i<ww.size(); i++)
		write_w(ww[i]._v, i);

//	test(dataset.test_images);

	return 0;
}


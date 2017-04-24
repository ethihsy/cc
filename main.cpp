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
	srand(time(NULL));
	//check_data(dataset.training_images, 11);

	vector<Layer> nn;
	nn.push_back(Layer(h_d, v_d+1));		//	wxh1
	nn.back().init_w();
	nn.push_back(Layer(batch, h_d));		//	h1
	nn.back().init_s(h_d, h_d+1);
	nn.push_back(Layer(h_d, h_d+1));		//	wh1h2
	nn.back().init_w();
	nn.push_back(Layer(batch, h_d));		//	h2
	nn.back().init_s(v_d, h_d+1);
	nn.push_back(Layer(v_d, h_d+1));		//	wh2y
	nn.back().init_w();
	nn.push_back(Layer(batch, v_d));		//	y

	double x[batch][v_d+1];
	int max_itr = dataset.training_images.size()/batch;
	double loss[batch][v_d];
	double gg[max_itr], gd[3];
	double ll[max_itr];

	fstream fs;
	fs.open("train_loss.txt", ios::out);

//	------------------------------------------------------------------------------	

	for (int epoch=0; epoch<max_epoch; epoch++){
		shuffle(begin(dataset.training_images), end(dataset.training_images), default_random_engine(time(NULL)));
		memset(ll, 0, sizeof(ll));
	
		for (int itr=0; itr<dataset.training_images.size(); itr+=batch){
			for (int i=1; i<nn.size(); i+=2)
				nn[i].reset();
			for (int i=0; i<nn.size(); i+=2)
				nn[i].reset_a(0);
			nn.back().reset_a(1);

			for (int i=0; i<batch; i++){
				x[i][v_d] = 1;
				for (int j=0; j<v_d; j++){
					x[i][j] = (double)dataset.training_images[i+itr][j];
					loss[i][j] = (double)dataset.training_images[i+itr][j+v_d];
				}
			}

			feed_forward(x, nn[0]._v, nn[1]._v);
			for (int i=1; i<5; i+=2){
				sample(nn[i]._v, nn[i]._s);
				feed_forward(nn[i]._s, nn[i+1]._v, nn[i+2]._v, nn[i]._sy);
			}
			loss_function(nn.back()._v, nn.back()._a, loss);

			for (int i=nn.size()-1; i>2; i-=2){
				back_prop(nn[i]._a, nn[i]._v, nn[i-1]._a, nn[i-1]._v, nn[i-2]._as, nn[i-2]._s, nn[i-2]._asy, nn[i-2]._sy);
				a_sample(nn[i-2]._as, nn[i-2]._s, nn[i-2]._a, nn[i-2]._v, nn[i]._v);
			}
			back_prop_w(nn[1]._a, nn[1]._v, nn[0]._a, x);

			for (int i=0; i<=4; i+=2)
				optimizer(nn[i]._v, nn[i]._a, nn[i]._m);
			
	//		check_gd(2,2,4,nn,x,dataset.training_images,loss);

			for (int i=0; i<batch; i++)
				for (int j=0; j<v_d; j++)
					ll[itr/batch] += loss[i][j];
			ll[itr/batch] /= batch;

			memset(gd, 0, sizeof(gd));
			for (int i=0; i<h_d; i++){
				for (int j=0; j<v_d+1; j++)
					gd[0] += nn[0]._a[i][j] * nn[0]._a[i][j];
				for (int j=0; j<h_d+1; j++)
					gd[1] += nn[2]._a[i][j] * nn[2]._a[i][j];
			}
			for (int i=0; i<v_d; i++)
				for (int j=0; j<h_d+1; j++)
					gd[2] += nn[4]._a[i][j] * nn[4]._a[i][j];
			gg[itr/batch] = (sqrt(gd[0])+sqrt(gd[1])+sqrt(gd[2])) / 3.0;

		}
		for (int i=0; i<max_itr; i++)
			fs << i+epoch*max_itr << "\t" << ll[i] << endl;
		cout << "epoch: " << epoch << " is done" << endl;
		
	}
	fs.close();
	for (int i=0; i<=4; i+=2)
		write_w(nn[i]._v, i);

//	test(dataset.test_images);
//	------------------------------------------------------------------------------



	return 0;
}


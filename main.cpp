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
	auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>();
	binarize_dataset(dataset);
	srand(time(NULL));
	//check_data(dataset.training_images, 11);

	double x[batch][v_d+1];
	mtx_w<h_d, v_d> wxh1;	wxh1.init();
	mtx_s<h_d> h1;
	mtx_w<h_d, h_d> wh1h2;	wh1h2.init();
	mtx_s<v_d> h2;
	mtx_w<v_d, h_d> wh2y;	wh2y.init();
	mtx y;
	double loss[batch][v_d];

	int max_itr = dataset.training_images.size()/batch;
	double gg[max_itr], gd[3];
	double ll[max_itr];

	fstream fs;
	fs.open("train_loss.txt", ios::out);
typedef boost::variant<double[batch][v_d+1], mtx, mtx_s<h_d>, mtx_s<v_d>, 
		  					mtx_w<h_d, v_d>, mtx_w<h_d, h_d>, mtx_w<v_d, h_d>> nn;
nn tt = wh1h2, t1=h1, t2=h2;
//	------------------------------------------------------------------------------	

	for (int epoch=0; epoch<max_epoch; epoch++){
		shuffle(begin(dataset.training_images), end(dataset.training_images), default_random_engine(time(NULL)));
		memset(ll, 0, sizeof(ll));
	
		for (int itr=0; itr<dataset.training_images.size(); itr+=batch){
			wxh1.reset();	h1.reset();	wh1h2.reset();	h2.reset();	wh2y.reset();	y.reset();
			for (int i=0; i<batch; i++){
				x[i][v_d] = 1;
				for (int j=0; j<v_d; j++){
					x[i][j] = (double)dataset.training_images[i+itr][j];
					loss[i][j] = (double)dataset.training_images[i+itr][j+v_d];
				}
			}
		
			feed_forward(x, wxh1._v, h1._v);
			sample(h1._v, h1._s);
	//		s_feed_forward(h1._s, wh1h2._v, h2._v, h1._sy);
			s_feed_forward(t1, tt, t2);
			sample(h2._v, h2._s);
	//		s_feed_forward(h2._s, wh2y._v, y._v, h2._sy);

			loss_function(y._v, y._a, loss);
	
			s_back_prop(y._a, y._v, wh2y._a, wh2y._v, h2._as, h2._s, h2._asy, h2._sy);
			a_sample(h2._as, h2._s, h2._a, h2._v);
			s_back_prop(h2._a, h2._v, wh1h2._a, wh1h2._v, h1._as, h1._s, h1._asy, h1._sy);
			a_sample(h1._as, h1._s, h1._a, h1._v);
			back_prop_w(h1._a, h1._v, wxh1._a, x);

			optimizer(wxh1._v, wxh1._a, wxh1._m);
			optimizer(wh1h2._v, wh1h2._a, wh1h2._m);
			optimizer(wh2y._v, wh2y._a, wh2y._m);
	
	//		check_gd(2,2,4,nn,x,dataset.training_images,loss);

			for (int i=0; i<batch; i++)
				for (int j=0; j<v_d; j++)
					ll[itr/batch] += loss[i][j];
			ll[itr/batch] /= batch;

			memset(gd, 0, sizeof(gd));
			for (int i=0; i<h_d; i++){
				for (int j=0; j<v_d+1; j++)
					gd[0] += wxh1._a[i][j] * wxh1._a[i][j];
				for (int j=0; j<h_d+1; j++)
					gd[1] += wh1h2._a[i][j] * wh1h2._a[i][j];
			}
			for (int i=0; i<v_d; i++)
				for (int j=0; j<h_d+1; j++)
					gd[2] += wh2y._a[i][j] * wh2y._a[i][j];
			gg[itr/batch] = (sqrt(gd[0])+sqrt(gd[1])+sqrt(gd[2])) / 3;
	
		}
		for (int i=0; i<max_itr; i++)
			fs << i+epoch*max_itr << "\t" << ll[i] << endl;
		cout << "epoch: " << epoch << " is done" << endl;
	}
	fs.close();
//	for (int i=0; i<=4; i+=2)
//		write_w(nn[i]._v, i);

//	test(dataset.test_images);
//	------------------------------------------------------------------------------



	return 0;
}


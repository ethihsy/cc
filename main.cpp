#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
//#include <omp.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"
#include "f.hpp"

using namespace std;
//using namespace cv;

class Net{
	public:
		Net(int m, int n) : _v(m, vector<double>(n)), _a(_v), w(true){}
		vector<vector<double>> _v;
		vector<vector<double>> _a;
		vector<vector<double>> _m;

		void init_w(int m, int n){
			for (int i=0; i<_v.size(); i++)
				for (int j=0; j<_v.front().size(); j++)
					_v[i][j] = (double)rand()/RAND_MAX - 0.5;
			_m.assign(m, vector<double>(n, 0));
			w = false;
		}
		void init(){
			if (w){
				for (int i=0; i<_v.size(); i++){
					fill(_v[i].begin(), _v[i].end(), 0);
					fill(_a[i].begin(), _a[i].end(), 0);
				}
			}
			else{
				for (int i=0; i<_v.size(); i++)
					fill(_a[i].begin(), _a[i].end(), 0);
			}
		}
	private:
		bool w;
};

int main(){
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	binarize_dataset(dataset);
	srand(time(NULL));
	//check_data(dataset.training_images, 11);

	const int v_dim=392, h_dim=200, ly=8;
	vector<Net> nn;
	nn.push_back(Net(h_dim, v_dim+1));	//	wxh1
	nn.back().init_w(h_dim, v_dim+1);
	nn.push_back(Net(batch, h_dim));	//	h1
	nn.push_back(Net(batch, h_dim+1));	//	sh1
	nn.push_back(Net(h_dim, h_dim+1));	//	wh1h2
	nn.back().init_w(h_dim, h_dim+1);
	nn.push_back(Net(batch, h_dim));	//	h2
	nn.push_back(Net(batch, h_dim+1));	//	sh2
	nn.push_back(Net(v_dim, h_dim+1));	//	wh2y
	nn.back().init_w(v_dim, h_dim+1);
	nn.push_back(Net(batch, v_dim));	//	y
	
	double x[batch][v_dim+1];
	int max_itr = dataset.training_images.size()/batch;
	double loss[batch][v_dim];
	double gg[max_itr], gd[3];
	double ll[max_itr];

	fstream fs;
	fs.open("train_loss.txt", ios::out);

//	------------------------------------------------------------------------------	

	for (int epoch=0; epoch<max_epoch; epoch++){
		shuffle(begin(dataset.training_images), end(dataset.training_images), default_random_engine(time(NULL)));
		memset(ll, 0, sizeof(ll));
		for (int itr=0; itr<dataset.training_images.size(); itr+=batch){
			for (int i=0; i<ly; i++)
				nn[i].init();
			for (int i=0; i<batch; i++){
				x[i][v_dim] = 1;
				nn[2]._v[i].back() = 1;
				nn[5]._v[i].back() = 1;
				for (int j=0; j<v_dim; j++){
					x[i][j] = (double)dataset.training_images[i+itr][j];
					nn.back()._a[i][j] = 1;
					loss[i][j] = (double)dataset.training_images[i+itr][j+v_dim];
				}
			}

			feed_forward(x, nn[0]._v, nn[1]._v);
			for (int i=1; i<5; i+=3){
				sample(nn[i]._v, nn[i+1]._v);
				feed_forward(nn[i+1]._v, nn[i+2]._v, nn[i+3]._v);
			}
			loss_function(nn.back()._v, nn.back()._a, loss);

			for (int i=ly-1; i>3; i-=3){
				back_prop_w(nn[i]._a, nn[i]._v, nn[i-1]._a, nn[i-2]._v);
				back_prop_x(nn[i]._a, nn[i]._v, nn[i-1]._v, nn[i-2]._a);
				a_sample(nn[i-2]._a, nn[i-2]._v, nn[i-3]._a, nn[i-3]._v, nn[i]._v);
			}
			back_prop_w(nn[1]._a, nn[1]._v, nn[0]._a, x);

			for (int i=0; i<=ly; i+=3)
				optimizer(nn[i]._v, nn[i]._a, nn[i]._m);
			
			for (int i=0; i<batch; i++)
				for (int j=0; j<v_dim; j++)
					ll[itr/batch] += loss[i][j];
			ll[itr/batch] /= batch;
			memset(gd, 0, sizeof(gd));
			for (int i=0; i<h_dim; i++){
				for (int j=0; j<v_dim+1; j++)
					gd[0] += nn[0]._a[i][j] * nn[0]._a[i][j];
				for (int j=0; j<h_dim+1; j++)
					gd[1] += nn[3]._a[i][j] * nn[3]._a[i][j];
			}
			for (int i=0; i<v_dim; i++)
				for (int j=0; j<h_dim+1; j++)
					gd[2] += nn[6]._a[i][j] * nn[6]._a[i][j];
			gg[itr/batch] = (sqrt(gd[0])+sqrt(gd[1])+sqrt(gd[2])) / 3;

		}
		for (int i=0; i<max_itr; i++)
			fs << i+epoch*max_itr << "\t" << ll[i] << endl;
		cout << "epoch: " << epoch << " is done" << endl;
	}
	fs.close();
	for (int i=0; i<=6; i+=3)
		write_w(nn[i]._v, i);
//	------------------------------------------------------------------------------
/*
	if (test){
		for (int i=0; i<dataset.test_images.size(); i++)
			dataset.test_images[i].insert(dataset.test_images[i].begin(), 1);
		max_itr = dataset.test_images.size()/batch;
		double llv[max_itr];
		memset(llv, 0, sizeof(llv));
		fs.open("test_loss.txt", ios::out);
		fs << "itr\tloss/step" << endl;
		for (int itr=0; itr<dataset.test_images.size(); itr+=batch){




			for (int i=0; i<batch; i++)
				for (int j=0; j<v_dim; j++)
					llv[itr/batch] += loss[i][j];
			llv[itr/batch] /= batch;
		}
		for (int i=0; i<max_itr; i++)
			fs << i << "\t" << llv[i] << endl;
		fs.close();
	}
*/
//	------------------------------------------------------------------------------

	/*
	Mat img(28, 28, CV_8U);
	for (int i=0; i<28; i++)
		for (int j=0; j<28; j++)
			img2.at<uchar>(i,j) = dataset.test_images[0][j+i*28];
	imshow("222",img2);
*/


	return 0;
}


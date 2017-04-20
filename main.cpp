#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
#include <omp.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"
#include "f.hpp"

using namespace std;
//using namespace cv;

int main(){
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	binarize_dataset(dataset);
	//check_data(dataset.training_images, 11);
	const int v_dim=392, h_dim=200;
	double h1[batch][h_dim], h2[batch][h_dim], y[batch][v_dim], loss[batch][v_dim];
	double s_h1[batch][n_s][h_dim+1], s_h2[batch][n_s][h_dim+1];
	double r_h1[batch][n_s][h_dim], r_h2[batch][n_s][v_dim];
	double w_xh1[h_dim][v_dim+1], w_h1h2[h_dim][h_dim+1], w_h2y[v_dim][h_dim+1];

	double as_h[batch][h_dim+1], ah1[batch][h_dim], ah2[batch][h_dim], ay[batch][v_dim];
	double aw_xh1[h_dim][v_dim+1], aw_h1h2[h_dim][h_dim+1], aw_h2y[v_dim][h_dim+1];
	double vaw_xh1[h_dim][v_dim+1], vaw_h1h2[h_dim][h_dim+1], vaw_h2y[v_dim][h_dim+1];

	int max_itr = dataset.training_images.size()/batch;
	double ll[max_itr];
	double gg[max_itr], ggt[3];
	fstream file;
	file.open("train_loss.txt", ios::out);

	memset( vaw_xh1, 0, sizeof(vaw_xh1));
	memset(vaw_h1h2, 0, sizeof(vaw_h1h2));
	memset( vaw_h2y, 0, sizeof(vaw_h2y));
	srand(time(NULL));
//	------------------------------------------------------------------------------	
#pragma omp parallel
{
	#pragma omp for
	for (int i=0; i<dataset.training_images.size(); i++)
		dataset.training_images[i].insert(dataset.training_images[i].begin(), 1);
	#pragma omp for
	for (int i=0;i<h_dim;i++){
		for (int j=0;j<v_dim+1;j++)
			w_xh1[i][j] = 0.1; //(double)rand()/RAND_MAX - 0.5;
		for (int j=0;j<h_dim+1;j++)
			w_h1h2[i][j] = 0.1; //(double)rand()/RAND_MAX - 0.5;
	}
	#pragma omp for
	for (int i=0;i<v_dim;i++)
		for (int j=0;j<h_dim+1;j++)
			w_h2y[i][j]  = 0.1; //(double)rand()/RAND_MAX - 0.5;

	for (int epoch=0; epoch<max_epoch; epoch++){
		#pragma omp single
		{
			//shuffle(begin(dataset.training_images), end(dataset.training_images), default_random_engine(time(NULL)));
			memset(ll, 0, sizeof(ll));
			memset(gg, 0, sizeof(gg));
		}
		for (int itr=0; itr<1; itr++){
//		for (int itr=0; itr<dataset.training_images.size(); itr+=batch){
			#pragma omp single
			{
				memset(h1, 0, sizeof(h1));	memset(h2, 0, sizeof(h2));	memset(y, 0, sizeof(y));
				memset(aw_xh1, 0, sizeof(aw_xh1));	memset(aw_h1h2, 0, sizeof(aw_h1h2));
				memset(aw_h2y, 0, sizeof(aw_h2y));
				memset(ah2, 0, sizeof(ah2));	memset(ah1, 0, sizeof(ah1));	memset(as_h, 0, sizeof(as_h));
			}
			#pragma omp for
			for (int i=0; i<batch; i++){
				for (int j=0; j<v_dim; j++){
					ay[i][j] = 1;
					loss[i][j] = (double)dataset.training_images[i+itr][j+v_dim+1];
				}
			}
			#pragma omp for reduction(+:aw_xh1, aw_h1h2, aw_h2y)
			for (int i=0; i<batch; i++){
				feed_forward(dataset.training_images[itr+i], w_xh1, h1[i], h1[i], false);
				for (int j=0; j<n_s; j++){
					sample(h1[i], s_h1[i][j]);
					feed_forward(s_h1[i][j], w_h1h2, r_h1[i][j], h2[i], true);
				}
				for (int j=0; j<n_s; j++){
					sample(h2[i], s_h2[i][j]);
					feed_forward(s_h2[i][j], w_h2y, r_h2[i][j], y[i], true);
				}
				cout << r_h2[0][0][44] << " " << y[0][44] << " " << s_h2[0][0][144] << endl;
		//		loss_function(y[i], loss[i], ay[i]);
				for (int j=0; j<n_s; j++){
					back_prop(s_h2[i][j], w_h2y, r_h2[i][j], ay[i], aw_h2y, true, as_h[i]);
					a_sample(h2[i], s_h2[i][j], as_h[i], ah2[i], y[i]);	
				}
				cout << aw_h2y[44][144] << " " << s_h2[0][0][144]*r_h2[0][0][44]*(1-r_h2[0][0][44]) << endl;
				for (int j=0; j<n_s; j++){
					back_prop(s_h1[i][j], w_h1h2, r_h1[i][j], ah2[i], aw_h1h2, true, as_h[i]);
					a_sample(h1[i], s_h1[i][j], as_h[i], ah1[i], h2[i]);
				}
				back_prop(dataset.training_images[itr+i], w_xh1, h1[i], ah1[i], aw_xh1, false, dataset.training_images[itr+i]);
			}
			#pragma omp sections
			{
				#pragma omp section
					optimizer(w_h2y, aw_h2y, vaw_h2y, h_dim+1, v_dim);
				#pragma omp section
					optimizer(w_h1h2, aw_h1h2, vaw_h1h2, h_dim+1, h_dim);
				#pragma omp section
					optimizer(w_xh1, aw_xh1, vaw_xh1, v_dim+1, h_dim);
			}
			#pragma omp single
			{
			for (int i=0; i<batch; i++)
				for (int j=0; j<v_dim; j++)
					ll[itr/batch] += loss[i][j];
			ll[itr/batch] /= batch;

			memset(ggt,0,sizeof(ggt));
			for (int i=0; i<h_dim; i++){
				for (int j=0; j<v_dim+1; j++)
					ggt[0] += aw_xh1[i][j]*aw_xh1[i][j];
				for (int j=0; j<h_dim+1; j++)
					ggt[1] += aw_h1h2[i][j]*aw_h1h2[i][j];
			}
			for (int i=0; i<v_dim; i++)
				for (int j=0; j<h_dim+1; j++)
					ggt[2] += aw_h2y[i][j]*aw_h2y[i][j];
			for (int i=0; i<3; i++)
				gg[itr/batch] += sqrt(ggt[i])/3.;
			}
		}
		#pragma omp single
		{
		for (int i=0; i<max_itr; i++)
			file << i+epoch*max_itr << "\t" << ll[i] << "\t" << gg[i] << endl;
		cout << "epoch: " << epoch << " is done" << endl;
		}
	}
}
	file.close();
	write_w(w_xh1,	h_dim, v_dim+1, "w_xh1.txt");
	write_w(w_h1h2, h_dim, h_dim+1, "w_h1h2.txt");
	write_w(w_h2y,	v_dim, h_dim+1, "w_h2y.txt");

//	------------------------------------------------------------------------------
/*
	for (int i=0; i<dataset.test_images.size(); i++)
		dataset.test_images[i].insert(dataset.test_images[i].begin(), 1);
	max_itr = dataset.test_images.size()/batch;
	double llv[max_itr];	memset(llv, 0, sizeof(llv));
	file.open("test_loss.txt", ios::out);
	file << "itr\tloss/step" << endl;
	for (int itr=0; itr<dataset.test_images.size(); itr+=batch){
		memset(h1, 0, sizeof(h1));	memset(h2, 0, sizeof(h2));	memset(y, 0, sizeof(y));
		for (int i=0; i<batch; i++)
			for (int j=0; j<v_dim; j++)
				loss[i][j] = (double)dataset.training_images[i+itr][j+v_dim+1];

		for (int i=0; i<batch; i++){
			feed_forward(dataset.test_images[itr+i], w_xh1, h1[i], h1[i], false);
			sample(h1[i], s_h1[i]);
			for (int j=0; j<n_s; j++)
				feed_forward(s_h1[i][j], w_h1h2, r_h1[i][j], h2[i], true);
			sample(h2[i], s_h2[i]);
			for (int j=0; j<n_s; j++)
				feed_forward(s_h2[i][j], w_h2y, r_h2[i][j], y[i], true);
			loss_function(y[i], loss[i], ay[i]);
		}
		for (int i=0; i<batch; i++)
			for (int j=0; j<v_dim; j++)
				llv[itr/batch] += loss[i][j];
		llv[itr/batch] /= batch;
	}
	for (int i=0; i<max_itr; i++)
		file << i << "\t" << llv[i] << endl;
	file.close();
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


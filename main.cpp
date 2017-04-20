#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
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
	for (int i=0; i<dataset.training_images.size(); i++)
		dataset.training_images[i].insert(dataset.training_images[i].begin(), 1);

	const int v_dim=392, h_dim=200;
	double h1[batch][h_dim+1], h2[batch][h_dim+1], y[batch][v_dim], loss[batch][v_dim];
	double s_h1[n_s][batch][h_dim+1], s_h2[n_s][batch][h_dim+1];
	double r_h1[n_s][batch][h_dim+1], r_h2[n_s][batch][v_dim];
	for (int i=0; i<batch; i++)
		h1[i][h_dim] = h2[i][h_dim] = 1;

	double w_xh1[h_dim][v_dim+1],   w_h1h2[h_dim][h_dim+1],   w_h2y[v_dim][h_dim+1];
	srand(time(NULL));	
	for (int i=0;i<h_dim;i++)		for (int j=0;j<v_dim+1;j++)	w_xh1[i][j]  = (double)rand()/RAND_MAX - 0.5;
	for (int i=0;i<h_dim;i++)		for (int j=0;j<h_dim+1;j++)	w_h1h2[i][j] = (double)rand()/RAND_MAX - 0.5;
	for (int i=0;i<v_dim;i++)		for (int j=0;j<h_dim+1;j++)	w_h2y[i][j]  = (double)rand()/RAND_MAX - 0.5;

	double ar_h[batch][h_dim+1];
	double ah1[batch][h_dim+1],  ah2[batch][h_dim+1],   ay[batch][v_dim],
		aw_xh1[h_dim][v_dim+1],   aw_h1h2[h_dim][h_dim+1],   aw_h2y[v_dim][h_dim+1],
	   vaw_xh1[h_dim][v_dim+1],  vaw_h1h2[h_dim][h_dim+1],  vaw_h2y[v_dim][h_dim+1];
	memset( vaw_xh1, 0, sizeof(vaw_xh1));
	memset(vaw_h1h2, 0, sizeof(vaw_h1h2));
	memset( vaw_h2y, 0, sizeof(vaw_h2y));

//	------------------------------------------------------------------------------

	int max_itr = dataset.training_images.size()/batch;
	double ll[max_itr];
	double gg[max_itr];
	fstream file;
	file.open("train_loss.txt", ios::out);

	for (int epoch=0; epoch<max_epoch; epoch++){
		shuffle(begin(dataset.training_images), end(dataset.training_images), default_random_engine(time(NULL)));
		for (int itr=0; itr<dataset.training_images.size(); itr+=batch){
			memset(h1, 0, sizeof(h1));	memset(h2, 0, sizeof(h2));
			memset(y, 0, sizeof(y));
			memset(aw_xh1, 0, sizeof(aw_xh1));	memset(aw_h1h2, 0, sizeof(aw_h1h2));
			memset(aw_h2y, 0, sizeof(aw_h2y));
			memset(ah2, 0, sizeof(ah2));		memset(ah1, 0, sizeof(ah1));
			for (int i=0; i<batch; i++)	
				for (int j=0; j<v_dim; j++)	
					ay[i][j]=1;
			for (int i=0; i<batch; i++)
				for (int j=0; j<v_dim; j++)
					loss[i][j] = (double)dataset.training_images[i+itr][j+v_dim+1];

			feed_forward(dataset.training_images, itr, w_xh1, h1, h1, v_dim+1, h_dim, false);
			sample(h1, s_h1, h_dim);
			for (int i=0; i<n_s; i++)
				feed_forward(s_h1[i], 0, w_h1h2, r_h1[i], h2, h_dim+1, h_dim, true);
			sample(h2, s_h2, h_dim);
			for (int i=0; i<n_s; i++)
				feed_forward(s_h2[i], 0, w_h2y, r_h2[i], y, h_dim+1, v_dim, true);
			loss_function(y, loss, ay, v_dim);

			for (int i=0; i<n_s; i++){
				back_prop(s_h2[i], 0, w_h2y, r_h2[i], ay, aw_h2y, true, ar_h, vaw_h2y, h_dim+1, v_dim);
				a_sample(ar_h, s_h2[i], h2, ah2, y, h_dim);
			}	
			for (int i=0; i<n_s; i++){
				back_prop(s_h1[i], 0, w_h1h2, r_h1[i], ah2, aw_h1h2, true, ar_h, vaw_h1h2, h_dim+1, h_dim);
				a_sample(ar_h, s_h1[i], h1, ah1, h2, h_dim);
			}
			back_prop(dataset.training_images, itr, w_xh1, h1, ah1, aw_xh1, false, dataset.training_images, vaw_xh1, v_dim+1, h_dim);
			optimizer(w_h2y, aw_h2y, vaw_h2y, h_dim+1, v_dim);
			optimizer(w_h1h2, aw_h1h2, vaw_h1h2, h_dim+1, h_dim);
			optimizer(w_xh1, aw_xh1, vaw_xh1, v_dim+1, h_dim);

			ll[itr/batch] = 0;
			for (int i=0; i<batch; i++)
				for (int j=0; j<v_dim; j++)
					ll[itr/batch] += loss[i][j];
			ll[itr/batch] /= batch;

			double ggt[3];	memset(ggt,0,sizeof(ggt));
			for (int i=0; i<h_dim; i++)	for (int j=0; j<v_dim+1; j++)		ggt[0] += aw_xh1[i][j]*aw_xh1[i][j];
			for (int i=0; i<h_dim; i++)	for (int j=0; j<h_dim+1; j++)		ggt[1] += aw_h1h2[i][j]*aw_h1h2[i][j];
			for (int i=0; i<v_dim; i++)	for (int j=0; j<h_dim+1; j++)		ggt[2] += aw_h2y[i][j]*aw_h2y[i][j];
			gg[itr/batch] = (sqrt(ggt[0])+sqrt(ggt[1])+sqrt(ggt[2]))/3.;
		}
		for (int i=0; i<max_itr; i++)
			file << i+epoch*max_itr << "\t" << ll[i] << "\t" << gg[i] << endl;
		cout << "epoch: " << epoch << " is done" << endl;
	}
	file.close();
	write_w(w_xh1,	h_dim, v_dim+1, "w_xh1.txt");
	write_w(w_h1h2, h_dim, h_dim+1, "w_h1h2.txt");
	write_w(w_h2y,	v_dim, h_dim+1, "w_h2y.txt");

//	------------------------------------------------------------------------------

//	for (int i=0; i<dataset.test_images.size(); i++)
//		dataset.test_images[i].insert(dataset.test_images[i].begin(), 1);
//	max_itr = dataset.test_images.size()/batch;
//	double llv[max_itr];
//	file.open("test_loss.txt", ios::out);
//	file << "itr\tloss/step" << endl;
//	for (int itr=0; itr<dataset.test_images.size(); itr+=batch){
//		memset(h1, 0, sizeof(h1));			memset(h2, 0, sizeof(h2));		memset(y, 0, sizeof(y));
//		for (int i=0; i<batch; i++)
//			for (int j=0; j<v_dim; j++)
//				loss[i][j] = (double)dataset.test_images[i+itr][j+v_dim+1];
//		feed_forward(dataset.test_images, itr, w_xh1, h1, true, s_h1, v_dim+1, h_dim);
//		feed_forward(s_h1, 0, w_h1h2, h2, true, s_h2, h_dim+1, h_dim);
//		feed_forward(s_h2, 0, w_h2y, y, false, y, h_dim+1, v_dim);
//		loss_function(y, loss, ay, v_dim);
//		llv[itr/batch] = 0;
//		for (int i=0; i<batch; i++)		for (int j=0; j<v_dim; j++)		llv[itr/batch] += loss[i][j];
//		llv[itr/batch] /= batch;
//	}
//	for (int i=0; i<max_itr; i++)
//		file << i << "\t" << llv[i] << endl;
//	file.close();

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


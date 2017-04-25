#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
#include <cstring>
#include <iomanip>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
using namespace std;
//using namespace cv;

const int v_d = 392;
const int h_d = 200;
const int n_s = 1;
enum _sample{pass, gumbel, LR};
_sample m_s= pass;

const int max_epoch = 1;
const int batch = 1;
double alpha = 3e-5;
double _gamma = 0.9;

typedef vector<vector<double>> MTX;

class Layer{
	public:
		Layer(int m, int n) : _v(m, vector<double>(n)), _a(m, vector<double>(n)){}
		void reset(){
			for (int i=0; i<_v.size(); i++){
				fill(_v[i].begin(), _v[i].end(), 0);
				fill(_a[i].begin(), _a[i].end(), 0);
			}
		}
		void seed(){
			for (int i=0; i<_a.size(); i++)
				fill(_a[i].begin(), _a[i].end(), 1);
		}
		MTX _v, _a;
};
class Weights{
	public:
		Weights(int m, int n):	_v(m, vector<double>(n)), _a(m, vector<double>(n)), _m(m, vector<double>(n, 0)){}
		void init(){
			for (int i=0; i<_v.size(); i++){
				fill(_m[i].begin(), _m[i].end(), 0);
				for (int j=0; j<_v.front().size(); j++){
					_v[i][j] = (double)rand()/RAND_MAX - 0.5;

				}
			}
		}
		void reset(){
			for (int i=0; i<_a.size(); i++)
					fill(_a[i].begin(), _a[i].end(), 0);
		}
		double norm(){
			double a = 0;
			for (int i=0; i<_a.size(); i++)
				for (int j=0; j<_a.front().size(); j++)
					a += _a[i][j] * _a[i][j];
			return sqrt(a);
		}
		MTX _v, _a, _m;
};
void sample(MTX &h, MTX &sh){
	switch(m_s){
		case 0:
			for (int k=0; k<batch; k++){
				for (int j=0; j<n_s; j++){
					for (int i=0; i<h_d; i++)
						sh[k][i+j*(h_d+1)] = h[k][i];
					sh[k][(j+1)*(h_d+1)-1] = 1;
				}
			}
			break;
		case 1:
			double u, p1, p0;
			for (int k=0; k<batch; k++){
				for (int j=0; j<n_s; j++){
					for (int i=0; i<h_d; i++){
						u = -log( -log((double)rand()/RAND_MAX));	p1 = exp( log(h[k][i])+u );
						u = -log( -log((double)rand()/RAND_MAX));	p0 = exp( log(1.0-h[k][i])+u );
						sh[k][i+j*(h_d+1)] = p1 / (p1+p0);
					}
					sh[k][(j+1)*(h_d+1)-1] = 1;
				}
			}
			break;
		case 2:
			for (int k=0; k<batch; k++){
				for (int j=0; j<n_s; j++){
					for (int i=0; i<h_d; i++)
						sh[k][i+j*(h_d+1)] = (double)rand()/RAND_MAX > h[k][i] ? 1 : 0;
					sh[k][(j+1)*(h_d+1)-1] = 1;
				}
			}
	}
}
void feed_forward(MTX &x, MTX &w, MTX &y){
	for (int k=0; k<batch; k++){
		for (int i=0; i<w.size(); i++){
			for (int j=0; j<w.front().size(); j++)
				y[k][i] += x[k][j] * w[i][j];
			y[k][i] = 1.0 / (1.0+exp(-y[k][i]));
		}
	}
}
void feed_forward(MTX &x, MTX &w, MTX &y, MTX &sy){
	int m = w.size();
	int n = w.front().size();

	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			for (int l=0; l<n_s; l++){
				for (int j=0; j<n; j++)
					sy[k][i+l*m] += x[k][j+l*n] * w[i][j];
				sy[k][i+l*m] = 1.0 / (1.0+exp(-sy[k][i+l*m]));
				y[k][i] += sy[k][i+l*m]/n_s;
			}
		}
	}
}
void loss_function(MTX &y, MTX &ay, MTX &l){
	for (int k=0; k<batch; k++){
		for (int i=0; i<y.front().size(); i++){
			ay[k][i] = -ay[k][i] * (l[k][i]/y[k][i] - (1.0-l[k][i])/(1.0-y[k][i]));
			l[k][i] = -(l[k][i]*log(y[k][i]) + (1.0-l[k][i])*log(1.0-y[k][i]));
		}
	}
}
void back_prop_w(MTX &ay, MTX &y, MTX &aw, MTX &x){
	for (int k=0; k<batch; k++){
		for (int i=0; i<aw.size(); i++){
			ay[k][i] *= y[k][i]*(1.0-y[k][i]);
			for (int j=0; j<aw.front().size(); j++)
				aw[i][j] += ay[k][i] * x[k][j];
		}
	}
}
void back_prop(MTX &ay, MTX &y, MTX &asy, MTX &sy, MTX &aw, MTX &w, MTX &ax, MTX &x){
	int m = aw.size();
	int n = aw.front().size();

	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			for (int l=0; l<n_s; l++){
				asy[k][i+l*m] = ay[k][i]/n_s * sy[k][i+l*m]*(1.0-sy[k][i+l*m]);
				for (int j=0; j<n; j++){
					aw[i][j] += asy[k][i+l*m] * x[k][j+l*n];
					ax[k][j+l*n] += asy[k][i+l*m] * w[i][j];
				}
			}
		}
	}
}
void a_sample(MTX &ash, MTX &sh, MTX &ah, MTX &h, MTX &y){
	switch(m_s){
		case 0:
			for (int k=0; k<batch; k++)
				for (int j=0; j<n_s; j++)
					for (int i=0; i<h_d; i++)
						ah[k][i] += ash[k][i+j*(h_d+1)];
			break;
		case 1:
			for (int k=0; k<batch; k++)
				for (int j=0; j<n_s; j++)
					for (int i=0; i<h_d; i++)
						ah[k][i] += ash[k][i+j*(h_d+1)] * sh[k][i+j*(h_d+1)]*(1.0-sh[k][i+j*(h_d+1)])/h[k][i];
			break;
		case 2:
			for (int k=0; k<batch; k++)
				for (int j=0; j<n_s; j++)
					for (int i=0; i<h_d; i++)
						ah[k][i] *= y[k][i] / h[k][i];
	}
}
void optimizer(MTX &w, MTX &aw, MTX &mw){
	for (int i=0; i<w.size(); i++){
		for (int j=0; j<w.front().size(); j++){
			aw[i][j] /= batch;
			mw[i][j] = _gamma*mw[i][j] + alpha*aw[i][j];
			w[i][j] = w[i][j] - mw[i][j];
		}
	}
}
void write_w(MTX &w, int n){
	string c;
	switch(n){
		case 0:
			c="w_xh1.txt";
			break;
		case 1:
			c="w_h1h2.txt";
			break;
		case 2:
			c="w_h2y.txt";
	}
	ofstream fs(c);
	for (int i=0; i<w.size(); i++)
		for (int j=0; j<w.front().size(); j++)
			fs << setprecision(17) << fixed << w[i][j] << " ";
	fs.close();
}
void read_w(MTX &w, int n, int m, string c){
	char buf[n*m*50], *s;
	ifstream fs(c);
	fs.read(buf, sizeof(buf));
	
	s = strtok(buf," ");
	for (int i=0; i<n; i++){
		for (int j=0; j<m; j++){
			w[i][j] = atof(s);
			s = strtok(NULL," ");
		}
	}
	fs.close();
}
template<typename T>
void check_data(T &x, int q){
	for (int j=0; j<28; j++){
		for (int i=0; i<28; i++)
			cout << (int)x[q][j*28+i] << " ";
		cout << endl;
	}
}
void check_gd(int lev, int pi, int pj, vector<Layer> &nn, vector<Weights> &ww, MTX &x, MTX &data, MTX &loss){
	cout << ww[lev]._a[pi][pj] << endl;;
	double hh=1e-6;
	double yy[v_d];	
	for (int i=0; i<v_d; i++)	
		yy[i]=loss[0][i];
//	------------------------------------------------------------------------------	
	for (int i=0; i<nn.size(); i++)
		nn[i].reset();
	nn.back().seed();
	for (int i=0; i<ww.size(); i++)
		ww[i].reset();
	for (int i=0; i<batch; i++){
		x[i][v_d] = 1;
		for (int j=0; j<v_d; j++)
			loss[i][j] = (double)data[i][j+v_d];
	}
	ww[lev]._v[pi][pj] -= hh;
	feed_forward(x, ww[0]._v, nn[0]._v);
	for (int i=0; i<5; i+=3){
		sample(nn[i]._v, nn[i+1]._v);
		feed_forward(nn[i+1]._v, ww[(i+3)/3]._v, nn[i+3]._v, nn[i+2]._v);
	}
	loss_function(nn.back()._v, nn.back()._a, loss);
//	------------------------------------------------------------------------------	
	if (lev==ww.size()-1)
		cout << (yy[pi]-loss[0][pi]) / hh;
	else{
		for (int i=0; i<v_d; i++)
			yy[i]-=loss[0][i];
		for (int i=1; i<v_d; i++) 
			yy[0] += yy[i];
		cout << yy[0]/hh << endl;
	}
	char zz;	cin >> zz;
}
template<typename T>
void test(T &data, vector<Layer> &nn){
	int max_itr = data.size()/batch;
	double llv[max_itr];
	double loss[batch][v_d];
	memset(llv, 0, sizeof(llv));
	fstream fs;
	fs.open("test_loss.txt", ios::out);

	for (int itr=0; itr<data.size(); itr+=batch){




		for (int i=0; i<batch; i++)
			for (int j=0; j<v_d; j++)
				llv[itr/batch] += loss[i][j];
		llv[itr/batch] /= batch;
	}
	for (int i=0; i<max_itr; i++)
		fs << i << "\t" << llv[i] << endl;
	fs.close();
}

	/*
	Mat img(28, 28, CV_8U);
	for (int i=0; i<28; i++)
		for (int j=0; j<28; j++)
			img2.at<uchar>(i,j) = dataset.test_images[0][j+i*28];
	imshow("222",img2);
*/

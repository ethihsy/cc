#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
#include <cstring>
#include <boost/variant.hpp>
#include <boost/variant/multivisitors.hpp>
#include <boost/variant/variant_fwd.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
using namespace std;
//using namespace cv;

const int v_d = 392;
const int h_d = 200;
const int n_s = 1;
const int m_s = 1;

const int max_epoch = 1;
const int batch = 1;
double alpha = 3e-5;
double _gamma = 0.9;

struct mtx{
	double _v[batch][v_d];
	double _a[batch][v_d];

	void reset(){
		for (int i=0; i<batch; i++)
			for (int j=0; j<v_d; j++)
				_a[i][j] = 1;
		memset(_v, 0, sizeof(_v));
	}
};
template<int m>
struct mtx_s{
	double _v[batch][h_d];				double _a[batch][h_d];
	double _s[batch][n_s][h_d+1];		double _as[batch][n_s][h_d+1];
	double _sy[batch][n_s][m];			double _asy[batch][n_s][m];

	void reset(){
		memset(_v, 0, sizeof(_v));		memset(_a, 0, sizeof(_a));
		memset(_as, 0, sizeof(_as));	memset(_sy, 0, sizeof(_sy));
	}
};
template<int r, int c>
struct mtx_w{
	double _v[r][c+1];
	double _a[r][c+1];
	double _m[r][c+1];
	
	void init(){
		for (int i=0; i<r; i++)
			for (int j=0; j<c+1; j++)
				_v[i][j] = (double)rand()/RAND_MAX - 0.5;
		memset(_m, 0, sizeof(_m));
	}
	void reset(){
		memset(_a, 0, sizeof(_a));
	}
};


//	0_pass	1_gumbel	2_likelihood
template <typename T, typename Ts>
void sample(T &h, Ts &sh){
	switch(m_s){
		case 0:
			for (int k=0; k<batch; k++){
				for (int j=0; j<n_s; j++){
					for (int i=0; i<h_d; i++)
						sh[k][j][i] = h[k][i];
					sh[k][j][h_d] = 1;
				}
			}
			break;
		case 1:
			double u, p1, p0;
			for (int k=0; k<batch; k++){
				for (int j=0; j<n_s; j++){
					for (int i=0; i<h_d; i++){
						u = -log( -log((double)rand()/RAND_MAX));	p1 = exp( log(h[k][i])+u );
						u = -log( -log((double)rand()/RAND_MAX));	p0 = exp( log(1.-h[k][i])+u );
						sh[k][j][i] = p1 / (p1+p0);
					}
					sh[k][j][h_d] = 1;
				}
			}
			break;
		case 2:
			for (int k=0; k<batch; k++){
				for (int j=0; j<n_s; j++){
					for (int i=0; i<h_d; i++)
						sh[k][j][i] = (double)rand()/RAND_MAX > h[k][i] ? 1 : 0;
					sh[k][j][h_d] = 1;
				}
			}
			break;
		default:	cout << "sample wrong" << endl;
	}
}
template<typename Tx, typename Tw, typename Ty>
void feed_forward(Tx &x, Tw &w, Ty &y){
	int n=sizeof(w[0])/sizeof(double);
	int m=sizeof(w)/sizeof(double)/n;
	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			for (int j=0; j<n; j++)
				y[k][i] += x[k][j] * w[i][j];
			y[k][i] = 1.0 / (1.0+exp(-y[k][i]));
		}
	}
}
/*
template<typename Tx, typename Tw, typename Ty, typename Tsy>
void s_feed_forward(T &x, T &w, T &y, T &sy){
	int n=sizeof(w[0])/sizeof(double);
	int m=sizeof(w)/sizeof(double)/n;
	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			for (int l=0; l<n_s; l++){
				for (int j=0; j<n; j++)
					sy[k][l][i] += x[k][l][j] * w[i][j];
				sy[k][l][i] = 1.0 / (1.0+exp(-sy[k][l][i]));
				y[k][i] += sy[k][l][i]/n_s;
			}
		}
	}
}
*/

struct s_feed_forward: public boost::static_visitor<>{
	template<typename Tx, typename Tw, typename Ty>
	void operator()(Tx &x, Tw &w, Ty &y){
	int n=sizeof(w[0])/sizeof(double);
	int m=sizeof(w)/sizeof(double)/n;
	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			for (int l=0; l<n_s; l++){
				for (int j=0; j<n; j++)
					sy[k][l][i] += x[k][l][j] * w[i][j];
				sy[k][l][i] = 1.0 / (1.0+exp(-sy[k][l][i]));
				y[k][i] += sy[k][l][i]/n_s;
			}
		}
	}
}
template<typename T, typename Tl>
void loss_function(T &y, T &ay, Tl &l){
	int n=sizeof(y)/sizeof(double)/batch;
	for (int k=0; k<batch; k++){
		for (int i=0; i<n; i++){
			ay[k][i] = -ay[k][i] * (l[k][i]/y[k][i] - (1.0-l[k][i])/(1.0-y[k][i]));
			l[k][i] = -(l[k][i]*log(y[k][i]) + (1.0-l[k][i])*log(1.0-y[k][i]));
		}
	}
}
template<typename Tx, typename Tw, typename Ty>
void back_prop_w(Ty &ay, Ty &y, Tw &aw, Tx &x){
	int n=sizeof(aw[0])/sizeof(double);
	int m=sizeof(aw)/sizeof(double)/n;
	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			ay[k][i] *= y[k][i]*(1.0-y[k][i]);
			for (int j=0; j<n; j++)
				aw[i][j] += ay[k][i] * x[k][j];
		}
	}
}
template<typename Tx, typename Tw, typename Ty, typename Tsy>
void s_back_prop(Ty &ay, Ty &y, Tw &aw, Tw &w, Tx &ax, Tx &x, Tsy &asy, Tsy &sy){ 
	int n=sizeof(aw[0])/sizeof(double);
	int m=sizeof(aw)/sizeof(double)/n;
	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			for (int l=0; l<n_s; l++){
				asy[k][l][i] = ay[k][i] / n_s;
				asy[k][l][i] *= sy[k][l][i] * (1-sy[k][l][i]);
				for (int j=0; j<n; j++){
					aw[i][j] += asy[k][l][i] * x[k][l][j];
					ax[k][l][j] += asy[k][l][i] * w[i][j];
				}
			}
		}
	}
}
/*
template<typename Tx, typename Tw, typename Ty>
void s_back_prop(Ty &ay, Ty &y, Tw &aw, Tw &w, Tx &ax, Tx &x){ 
	int n=sizeof(aw[0])/sizeof(double);
	int m=sizeof(aw)/sizeof(double)/n;
	double ty, ts;
	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			for (int l=0; l<n_s; l++){
				ty = 0;
				ts = ay[k][i] / n_s;
				for (int j=0; j<n; j++)
					ty += x[k][l][j] * w[i][j];
				ty = 1. / (1.+exp(-ty));
				ts *= ty * (1-ty);
				for (int j=0; j<n; j++){
					aw[i][j] += ts * x[k][l][j];
					ax[k][l][j] += ts * w[i][j];
				}
			}
		}
	}
}
*/
template<typename Ts, typename T>
void a_sample(Ts &ash, Ts &sh, T &ah, T &h){
	switch(m_s){
		case 0:
			for (int k=0; k<batch; k++)
				for (int j=0; j<n_s; j++)
					for (int i=0; i<h_d; i++)
						ah[k][i] += ash[k][j][i]/n_s;
			break;
		case 1:
			for (int k=0; k<batch; k++)
				for (int j=0; j<n_s; j++)
					for (int i=0; i<h_d; i++)
						ah[k][i] += ash[k][j][i] * sh[k][j][i]*(1.-sh[k][j][i])/h[k][i];
			break;
			/*
		case 2:
			for (int k=0; k<batch; k++)
				for (int j=0; j<n_s; j++)
					for (int i=0; i<h_d; i++)
						ah[k][i] *= y[k][i] / h[k][i];
			break;
			*/
		default:	cout << "a_sample wrong" << endl;
	}
}
template<typename T>
void optimizer(T &w, T &aw, T &vaw){
	int n=sizeof(w[0])/sizeof(double);
	int m=sizeof(w)/sizeof(double)/n;
	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			aw[i][j] /= batch;
			vaw[i][j] = _gamma*vaw[i][j] + alpha*aw[i][j];
			w[i][j] = w[i][j] - vaw[i][j];
		}
	}
}
template<typename T>
void write_w(T &w, int nn){
	int n=sizeof(w[0])/sizeof(double);
	int m=sizeof(w)/sizeof(double)/n;
	fstream fs;
	string c;
	if (nn==0)
		c="w_xh1.txt";
	else if (n==2)
		c="w_h1h2.txt";
	else
		c="w_h2y.txt";
	fs.open(c, ios::out);
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
			fs << w[i][j] << " ";
	fs.close();
}
template<typename T>
void read_w(T &w, int n, int m, string c){
	char buf[n*m*50], *s;
	fstream fs;
	fs.open(c, ios::in);
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
/*
template<typename T, typename T2, typename T3>
void check_gd(int lev, int pi, int pj, vector<Layer> &nn, T &x, T2 &data, T3 &loss){
	cout << nn[lev]._a[pi][pj] << endl;;
	double hh=1e-6;
//	double yy=loss[0][pi];
	double yy[v_d];	for (int i=0; i<v_d; i++)	yy[i]=loss[0][i];

	for (int i=0; i<nn.size(); i++)
		nn[i].init();
	for (int i=0; i<batch; i++){
		x[i][v_d] = 1;
		for (int j=0; j<n_s; j++){
			nn[1]._s[i][j].back() = 1;
			nn[3]._s[i][j].back() = 1;
		}
		for (int j=0; j<v_d; j++){
			x[i][j] = (double)data[i][j];
			nn.back()._a[i][j] = 1;
			loss[i][j] = (double)data[i][j+v_d];
		}
	}
	nn[0].init_w(h_d, v_d+1);
	nn[2].init_w(h_d, h_d+1);
	nn[4].init_w(v_d, h_d+1);

	nn[lev]._v[pi][pj] -= hh;
	feed_forward(x, nn[0]._v, nn[1]._v);
	for (int i=1; i<5; i+=2){
		sample(nn[i]._v, nn[i]._s);
		s_feed_forward(nn[i]._s, nn[i+1]._v, nn[i+2]._v);
	}
	loss_function(nn.back()._v, nn.back()._a, loss);
	
	for (int i=0; i<v_d; i++)	yy[i] -= loss[0][i];
	for (int i=1; i<v_d; i++)	yy[0] += yy[i];
	
//	cout << (yy-loss[0][pi]) / hh;
	cout << yy[0]/hh << endl;
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
*/
	/*
	Mat img(28, 28, CV_8U);
	for (int i=0; i<28; i++)
		for (int j=0; j<28; j++)
			img2.at<uchar>(i,j) = dataset.test_images[0][j+i*28];
	imshow("222",img2);
*/

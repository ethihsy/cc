#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <fstream>
#include <cstring>
using namespace std;

double alpha = 3e-5;
double _gamma = 0.9;

const int max_epoch = 200;
const int batch = 4;
const int n_s = 1;
int m_s = 0;

template <typename T, typename Ts>
void sample(T &y, Ts &s, int m){
	switch(m_s){
		case 0:
			double u, p1, p0;
			for (int k=0; k<n_s; k++){
				for (int i=0; i<batch; i++){
				 	for (int j=0; j<m; j++){
						u = -log( -log((double)rand()/RAND_MAX));	p1 = exp( log(y[i][j])+u );
						u = -log( -log((double)rand()/RAND_MAX));	p0 = exp( log(1.-y[i][j])+u );
						s[k][i][j] = p1 / (p1+p0);
					}
					s[k][i][m] = 1;
				}
			}
			break;
		case 1:
			for (int k=0; k<n_s; k++){
				for (int i=0; i<batch; i++){
					for (int j=0; j<m; j++)
						s[k][i][j] = (double)rand()/RAND_MAX > y[i][j] ? 1 : 0;
					s[k][i][m] = 1;
				}
			}
			break;
		default:	cout << "sample wrong" << endl;
	}
}
template<typename T, typename T2>
void a_sample(T &ary, T &s, T &y, T &ay, T2 &yy, int m){
	switch(m_s){
		case 0:
			for (int i=0; i<batch; i++)
				for (int j=m-1; j>=0; j--)
					ay[i][j] += ary[i][j] * s[i][j]*(1.-s[i][j])/y[i][j];
			break;
		case 1:
			for (int i=0; i<batch; i++)
				for (int j=m-1; j>=0; j--)
					ay[i][j] *= yy[i][j] / y[i][j];
			break;
		default:	cout << "a_sample wrong" << endl;
	}
}
template<typename Tx, typename Tw, typename Ty>
void feed_forward(Tx &x, int itr, Tw &w, Ty &rx, Ty &y, int n, int m, bool s){
	memset(rx, 0, sizeof(rx));
	for (int k=0; k<batch; k++){
		for (int i=0; i<m; i++){
			for (int j=0; j<n; j++)
				rx[k][i] += x[k+itr][j] * w[i][j];
			rx[k][i] = 1. / (1.+exp(-rx[k][i]));
		}
		rx[k][m] = 1;
	}
	if (s){
		for (int k=0; k<batch; k++)
			for (int i=0; i<m; i++)
				y[k][i] += rx[k][i] / n_s;
	}
}
template<typename T>
void loss_function(T &y, T &l, T &ay, int m){
	for (int i=0; i<batch; i++){
		for (int j=0; j<m; j++){
			ay[i][j] = -ay[i][j] * (l[i][j]/y[i][j] - (1.-l[i][j])/(1.-y[i][j])) / n_s;
			l[i][j] = -(l[i][j]*log(y[i][j]) + (1.-l[i][j])*log(1.-y[i][j]));
		}
	}
}
template<typename T>
void optimizer(T &w, T &aw, T &vaw, int n, int m){
	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			aw[i][j] /= batch;
			vaw[i][j] = _gamma*vaw[i][j] + alpha*aw[i][j];
			w[i][j] = w[i][j] - vaw[i][j];
		}
	}
}
template<typename Tx, typename Tw, typename Ty>
void back_prop(Tx &x, int itr, Tw &w, Ty &y, Ty &ay, Tw &aw, bool bp, Tx &ax, Tw &vaw, int n, int m){
	double ar;
	for (int k=0; k<batch; k++){
		for (int i=m-1; i>=0; i--){
			ar = ay[k][i]*y[k][i] * (1.-y[k][i]);
			for (int j=n-1; j>=0; j--)
				aw[i][j] += ar * x[k+itr][j];
		}
	}
	if (bp){
		for (int k=0; k<batch; k++){
			for (int i=m-1; i>=0; i--){
				ar = ay[k][i]*y[k][i] * (1.-y[k][i]);
				for (int j=n-1; j>=0; j--)
					ax[k][j] +=  ar * w[i][j];
			}
		}
	}
}
template<typename T>
void write_w(T &w, int n, int m, string c){
	fstream fs;
	fs.open(c, ios::out);
	for (int i=0; i<n; i++)
		for (int j=0; j<m; j++)
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

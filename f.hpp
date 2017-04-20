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

const int max_epoch = 1;
const int batch = 1;
const int n_s = 1;
int m_s = 0;

template <typename T, typename Ts>
void sample(T &h, Ts &s){
	int m = sizeof(h)/8;
	switch(m_s){
		case 0:
			for (int i=0; i<m; i++)
				s[i] = h[i];
			s[m] = 1;
			break;
		case 1:
			double u, p1, p0;
			for (int i=0; i<m; i++){
				u = -log( -log((double)rand()/RAND_MAX));	p1 = exp( log(h[i])+u );
				u = -log( -log((double)rand()/RAND_MAX));	p0 = exp( log(1.-h[i])+u );
				s[i] = p1 / (p1+p0);
			}
			s[m] = 1;
			break;
		case 2:
				for (int j=0; j<m; j++)
					s[j] = (double)rand()/RAND_MAX > h[j] ? 1 : 0;
				s[m] = 1;
			break;
		default:	cout << "sample wrong" << endl;
	}
}
template<typename Th, typename Ts, typename T2>
void a_sample(Th &h, Ts &s, Ts &as, Th &ah, T2 &yy){
	int m = sizeof(ah)/8;
	switch(m_s){
		case 0:
			for (int i=0; i<m; i++)
				ah[i] += as[i];
			break;
		case 1:
			for (int i=0; i<m; i++)
				ah[i] += as[i] * s[i]*(1.-s[i])/h[i];
			break;
		case 2:
			for (int i=m-1; i>=0; i--)
				ah[i] *= yy[i] / h[i];
			break;
		default:	cout << "a_sample wrong" << endl;
	}
}
template<typename Tx, typename Tw, typename Ty>
void feed_forward(Tx &x, Tw &w, Ty &rx, Ty &y, bool s){
	memset(rx, 0, sizeof(rx));
	int n=sizeof(x)/8, m=sizeof(y)/8;
	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++)
			rx[i] += x[j] * w[i][j];
		rx[i] = 1. / (1.+exp(-rx[i]));
	}
	if (s){
		for (int i=0; i<m; i++)
			y[i] += rx[i] / (double)n_s;
	}
}
template<typename T>
void loss_function(T &y, T &l, T &ay){
	int m = sizeof(y)/8;
	for (int i=0; i<m; i++){
		ay[i] = -ay[i] * (l[i]/y[i] - (1.-l[i])/(1.-y[i])) / (double)n_s;
		l[i] = -(l[i]*log(y[i]) + (1.-l[i])*log(1.-y[i]));
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
void back_prop(Tx &x, Tw &w, Ty &rx, Ty &ay, Tw &aw, bool bp, Tx &ax){
	double ar, m=sizeof(ay)/8, n=sizeof(ax)/8;
	for (int i=0; i<m; i++){
		ar = ay[i] * rx[i]*(1.-rx[i]);
		for (int j=0; j<n; j++)
			aw[i][j] += ar * x[j];
	}
	if (bp){
		memset(&ax[0], 0, n*sizeof(ax[0]));
		for (int i=0; i<m; i++){
			ar = ay[i] * rx[i]*(1.-rx[i]);
			for (int j=0; j<n; j++)
				ax[j] +=  ar * w[i][j];
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

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
const int batch = 4;
const int n_s = 1;
int m_s = 1;
bool test=false;

template <typename T, typename Ts>
void sample(T &h, Ts &sh){
	int m = h.front().size();
	switch(m_s){
		case 0:
			for (int k=0; k<batch; k++){
				for (int i=0; i<m; i++)
					sh[k][i] = h[k][i];
				sh[k][m] = 1;
			}
			break;
		case 1:
			double u, p1, p0;
			for (int k=0; k<batch; k++){
				for (int i=0; i<m; i++){
					u = -log( -log((double)rand()/RAND_MAX));	p1 = exp( log(h[k][i])+u );
					u = -log( -log((double)rand()/RAND_MAX));	p0 = exp( log(1.-h[k][i])+u );
					sh[k][i] = p1 / (p1+p0);
				}
				sh[k][m] = 1;
			}
			break;
		case 2:
			for (int k=0; k<batch; k++){
				for (int i=0; i<m; i++)
					sh[k][i] = (double)rand()/RAND_MAX > h[k][i] ? 1 : 0;
				sh[k][m] = 1;
			}
			break;
		default:	cout << "sample wrong" << endl;
	}
}
template<typename Ts, typename T, typename Ty>
void a_sample(Ts &ash, Ts &sh, T &ah, T &h, Ty &y){
	switch(m_s){
		case 0:
			for (int k=0; k<batch; k++)
				for (int i=0; i<ah.front().size(); i++)
					ah[k][i] += ash[k][i];
			break;
		case 1:
			for (int k=0; k<batch; k++)
				for (int i=0; i<ah.front().size(); i++)
					ah[k][i] += ash[k][i] * sh[k][i]*(1.-sh[k][i])/h[k][i];
			break;
		case 2:
			for (int k=0; k<batch; k++)
				for (int i=0; i<ah.front().size(); i++)
					ah[k][i] *= y[k][i] / h[k][i];
			break;
		default:	cout << "a_sample wrong" << endl;
	}
}
template <typename Ts, typename Th>
void avg_s(Ts &rh, Th &h){
	for (int j=0; j<n_s; j++)
		for (int k=0; k<sizeof(h)/8; k++)
			h[k] += rh[j][k]/n_s;
}
template<typename Tx, typename Tw, typename Ty>
void feed_forward(Tx &x, Tw &w, Ty &y){
	for (int k=0; k<batch; k++){
		for (int i=0; i<w.size(); i++){
			for (int j=0; j<w.front().size(); j++)
				y[k][i] += x[k][j] * w[i][j];
			y[k][i] = 1. / (1.+exp(-y[k][i]));
		}
	}
}
template<typename T, typename Tl>
void loss_function(T &y, T &ay, Tl &l){
	for (int k=0; k<batch; k++){
		for (int i=0; i<y.front().size(); i++){
			ay[k][i] = -ay[k][i] * (l[k][i]/y[k][i] - (1.-l[k][i])/(1.-y[k][i]));
			l[k][i] = -(l[k][i]*log(y[k][i]) + (1.-l[k][i])*log(1.-y[k][i]));
		}
	}
}
template<typename Tx, typename Tw, typename Ty>
void back_prop_w(Ty &ay, Ty &y, Tw &aw, Tx &x){ 
	double ar;
	for (int k=0; k<batch; k++){
		for (int i=0; i<aw.size(); i++){
			ar = ay[k][i] * y[k][i]*(1.-y[k][i]);
			for (int j=0; j<aw.front().size(); j++)
				aw[i][j] += ar * x[k][j];
		}
	}
}
template<typename Tx, typename Tw, typename Ty>
void back_prop_x(Ty &ay, Ty y, Tw &w, Tx &ax){
	double ar;
	for (int k=0; k<batch; k++){
		for (int i=0; i<y.front().size(); i++){
			ar = ay[k][i] * y[k][i]*(1.-y[k][i]);
			for (int j=0; j<ax.front().size(); j++)
				ax[k][j] +=  ar * w[i][j];
		}
	}
}
template<typename T>
void optimizer(T &w, T &aw, T &vaw){
	for (int i=0; i<w.size(); i++){
		for (int j=0; j<w.front().size(); j++){
			aw[i][j] /= batch;
			vaw[i][j] = _gamma*vaw[i][j] + alpha*aw[i][j];
			w[i][j] = w[i][j] - vaw[i][j];
		}
	}
}
template<typename T>
void write_w(T &w, int n){
	fstream fs;
	string c;
	if (n==0)
		c="w_xh1.txt";
	else if (n==3)
		c="w_h1h2.txt";
	else
		c="w_h2y.txt";
	fs.open(c, ios::out);
	for (int i=0; i<w.size(); i++)
		for (int j=0; j<w.front().size(); j++)
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

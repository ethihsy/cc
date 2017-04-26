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
enum _sample{LR, DET, GUMBEL};
_sample m_s= LR;

const int max_epoch = 1;
const int batch = 1;
double alpha = 3e-5;
double _gamma = 0.9;
//   ------------------------------------------------------------------------------   
typedef vector<vector<double>> MTX;

class Layer{
   public:
      Layer(int m, int n): _v(m, vector<double>(n)),
                           _a(m, vector<double>(n)){}
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
      Weights(int m, int n):  _v(m, vector<double>(n)), _a(m, vector<double>(n)),
                              _m(m, vector<double>(n, 0)){}
      void init(){
         for (int i=0; i<_v.size(); i++){
            fill(_m[i].begin(), _m[i].end(), 0);
            for (int j=0; j<_v.front().size(); j++)
               _v[i][j] = (double)rand()/RAND_MAX - 0.5;
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
void sample(Layer &h, Layer &sh){
   switch(m_s){
      case LR:
         for (int k=0; k<batch; k++)
            for (int j=0; j<n_s; j++){
               for (int i=0; i<h_d; i++)
                  sh._v[k][i+j*(h_d+1)] = (double)rand()/RAND_MAX > h._v[k][i] ? 1 : 0;
               sh._v[k][(j+1)*(h_d+1)-1] = 1;
            }
         break;
      case DET:
         for (int k=0; k<batch; k++)
            for (int j=0; j<n_s; j++){
               for (int i=0; i<h_d; i++)
                  sh._v[k][i+j*(h_d+1)] = h._v[k][i];
               sh._v[k][(j+1)*(h_d+1)-1] = 1;
            }
         break;
      case GUMBEL:
         double u, p1, p0;
         for (int k=0; k<batch; k++){
            for (int j=0; j<n_s; j++){
               for (int i=0; i<h_d; i++){
                  u = -log( -log((double)rand()/RAND_MAX));   p1 = exp( log(h._v[k][i])+u );
                  u = -log( -log((double)rand()/RAND_MAX));   p0 = exp( log(1.0-h._v[k][i])+u );
                  sh._v[k][i+j*(h_d+1)] = p1 / (p1+p0);
               }
               sh._v[k][(j+1)*(h_d+1)-1] = 1;
            }
         }
         break;
   }
}
void feed_forward(MTX &x, Weights &w, Layer &sy, int ns=1){
   int m = w._v.size();
   int n = w._v.front().size();

   for (int k=0; k<batch; k++)
      for (int l=0; l<ns; l++)
         for (int i=0; i<m; i++){
            for (int j=0; j<n; j++)
               sy._v[k][i+l*m] += x[k][j+l*n] * w._v[i][j];
            sy._v[k][i+l*m] = 1.0 / (1.0+exp(-sy._v[k][i+l*m]));
         }
}
void expect(Layer &sy, Layer &y, int ns){
   int m = y._v.front().size();
   for (int k=0; k<batch; k++)
      for (int i=0; i<m; i++){
         for (int l=0; l<ns; l++)
            y._v[k][i] += sy._v[k][i+l*m];
         y._v[k][i] /= ns;
      }
}
void loss_function(Layer &y, MTX &l){
   for (int k=0; k<batch; k++)
      for (int i=0; i<y._v.front().size(); i++){
         y._a[k][i] = -y._a[k][i] * (l[k][i]/y._v[k][i] - (1.0-l[k][i])/(1.0-y._v[k][i]));
         l[k][i] = -(l[k][i]*log(y._v[k][i]) + (1.0-l[k][i])*log(1.0-y._v[k][i]));
      }
}
void back_prop_w(Layer &y, Weights &w, MTX &x){
   for (int k=0; k<batch; k++)
      for (int i=0; i<w._a.size(); i++){
         y._a[k][i] *= y._v[k][i]*(1.0-y._v[k][i]);
         for (int j=0; j<w._a.front().size(); j++)
            w._a[i][j] += y._a[k][i] * x[k][j];
      }
}
void back_prop(Layer &y, Layer &sy, Weights &w, Layer &sh){
   int m = w._a.size();
   int n = w._a.front().size();

   for (int k=0; k<batch; k++)
      for (int i=0; i<m; i++)
         for (int l=0; l<n_s; l++){
            sy._a[k][i+l*m] = y._a[k][i]/n_s * sy._v[k][i+l*m]*(1.0-sy._v[k][i+l*m]);
            for (int j=0; j<n; j++){
               w._a[i][j] += sy._a[k][i+l*m] * sh._v[k][j+l*n];
               sh._a[k][j+l*n] += sy._a[k][i+l*m] * w._v[i][j];
            }
         }
}
void a_sample(Layer &sh, Layer &h, Layer &sy){
   int m = sy._v.front().size()/n_s;
   switch(m_s){
      case LR:
         for (int k=0; k<batch; k++)
            for (int j=0; j<n_s; j++)
               for (int i=0; i<h_d; i++)
                  for (int l=0; l<m; l++)
                     h._a[k][i] += sy._v[k][l+j*m] / (h._v[k][i]+1e-6) / n_s * h._a[k][i];
         break;      
      case DET:
         for (int k=0; k<batch; k++)
            for (int j=0; j<n_s; j++)
               for (int i=0; i<h_d; i++)
                  h._a[k][i] += sh._a[k][i+j*(h_d+1)];
         break;
      case GUMBEL:
         for (int k=0; k<batch; k++)
            for (int j=0; j<n_s; j++)
               for (int i=0; i<h_d; i++)
                  h._a[k][i] += sh._a[k][i+j*(h_d+1)] * sh._v[k][i+j*(h_d+1)]*(1.0-sh._v[k][i+j*(h_d+1)])/h._v[k][i];
         break;
   }
}
void optimizer(Weights &w){
   for (int i=0; i<w._v.size(); i++)
      for (int j=0; j<w._v.front().size(); j++){
         w._a[i][j] /= batch;
         w._m[i][j] = _gamma * w._m[i][j] + alpha * w._a[i][j];
         w._v[i][j] = w._v[i][j] - w._m[i][j];
      }
}

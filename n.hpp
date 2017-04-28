#include <armadillo>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
using namespace arma;
//using namespace cv;

const int max_epoch = 10;
const int batch = 40;
const int n_s = 1;
enum _sample{LR, DET};
_sample m_s= DET;

double alpha = 3e-5;
double _gamma = 0.9;

class Layer{
   public:
      Layer(int m, int n): v(m,n), a(m,n){}
      void reset(){
         v.zeros(); 
         a.zeros();
      }
      mat v, a;
};
class S_Layer{
   public:
      S_Layer(int m, int n, int l): v(l, mat(m,n)), a(l, mat(m,n)){}
      void reset(){
         for (int i=0; i<v.size(); i++)
            a[i].zeros();
      }
      std::vector<mat> v, a;
};
class Weights{
   public:
      Weights(int m, int n): v(n,m), a(n,m), mt(n,m){}
      void init(){
         mt.zeros();
         v.randu();
         v -= 0.5;
 /*        for (int i=0; i<v.n_rows; i++)
            for (int j=0; j<v.n_cols; j++)
               v(i,j) = (double)rand()/RAND_MAX - 0.5;*/
      }
      mat v, a, mt;
};
void sample(Layer &h, S_Layer &sh){
   switch(m_s){
      case LR:
         for (int i=0; i<n_s; i++){
            mat rd(size(h.v), fill::randu);
            sh.v[i].submat(0, 0, h.v.n_rows-1, h.v.n_cols-1) = conv_to<mat>::from(rd > h.v);
            sh.v[i].col(h.v.n_cols).ones();
         }
         break;
      case DET:
         for (int i=0; i<sh.v.size(); i++){
            sh.v[i].submat(0, 0, h.v.n_rows-1, h.v.n_cols-1) = h.v;
            sh.v[i].col(h.v.n_cols).ones();
         }
         break;
   }
}
void feed_forward(mat &x, Weights &w, Layer &y){
   y.v = x * w.v;
   y.v.transform([](double y){ return 1.0/(1.0+exp(-y)); });
}
void feed_forward(S_Layer &x, Weights &w, S_Layer &y){
   for (int i=0; i<x.v.size(); i++){
      y.v[i] = x.v[i] * w.v;
      y.v[i].transform([](double y){ return 1.0/(1.0+exp(-y)); });
   }
}
void expect(S_Layer &sy, Layer &y){
   for (int i=0; i<sy.v.size(); i++)
      y.v += sy.v[i];
   y.v /= sy.v.size();
}
void loss_function(Layer &y, mat &loss){
   y.a %= -(loss/y.v - (1.0-loss)/(1.0-y.v));
   loss = -(loss % log(y.v) + (1.0-loss) % log(1.0-y.v));
}
void back_prop_w(Layer &y, Weights &w, mat &x){
   y.a %= y.v % (1.0-y.v);
   w.a += x.t() * y.a;
}
void back_prop(Layer &y, S_Layer &sy, Weights &w, S_Layer &sh){
   for (int i=0; i<sy.a.size(); i++){
      sy.a[i] = y.a / n_s % sy.v[i] % (1.0-sy.v[i]);
      w.a += sh.v[i].t() * sy.a[i];
      sh.a[i] += (sy.a[i] * w.v.t());
   }
}
void a_sample(S_Layer &sh, Layer &h, S_Layer &sy, Layer &y){
   switch(m_s){
      case LR:
         for (int i=0; i<sh.a.size(); i++)
            h.a += y.a % sy.v[i] / h.v / n_s;
         break;      
      case DET:
         for (int i=0; i<sh.a.size(); i++)
            h.a += sh.a[i].submat(0, 0, h.a.n_rows-1, h.a.n_cols-1);
         break;
   }
}
void optimizer(Weights &w){
   w.a /= batch;
   w.mt = _gamma * w.mt + alpha * w.a;
   w.v -= w.mt;
}

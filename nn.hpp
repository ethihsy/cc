#include <iostream>
#include <fstream>
#include <armadillo>
#include <iomanip>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace arma;
//using namespace cv;

double alpha = 3e-5;
double _gamma = 0.9;

class Layer{
    public:
        Layer(int m, int n): v(m, n), a(m, n), s(m, n+1), as(m, n+1){}
        void reset(){
            v.zeros(); 
            a.zeros();
            as.zeros();
        }
        mat v, a, s, as;
};
class Weights{
    public:
        Weights(int m, int n): v(n+1, m), a(n+1, m), mt(n+1, m){}
        void init(){
            mt.zeros();
            v.randu();
            v -= 0.5;
        }
        mat v, a, mt;
};
void feed_forward(mat &x, Weights &w, Layer &y){
    y.v = x * w.v;
    y.v = 1.0 / (1.0+exp(-y.v));
}
void loss_function(Layer &y, mat &loss){
    y.a %= -(loss/y.v - (1.0-loss)/(1.0-y.v));
    loss = -(loss % log(y.v) + (1.0-loss) % log(1.0-y.v));
}
void back_prop_w(Layer &y, Weights &w, mat &x){
    y.a %= y.v % (1.0-y.v);
    w.a += x.t() * y.a;
}
void optimizer(Weights &w, int batch){
    w.a /= batch;
    w.mt = _gamma * w.mt + alpha * w.a;
    w.v -= w.mt;
}
void write_w(Weights &w, int n){
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
    for (int i=0; i<w.v.n_rows; i++)
        for (int j=0; j<w.v.n_cols; j++)
            fs << setprecision(17) << fixed << w.v(i, j) << " ";
    fs.close();
}
void read_w(Weights &w, string c){
    char buf[w.v.n_rows * w.v.n_cols * 50], *s;
    ifstream fs(c);
    fs.read(buf, sizeof(buf));
    
    s = strtok(buf, " ");
    for (int i=0; i<w.v.n_rows; i++)
        for (int j=0; j<w.v.n_cols; j++){
            w.v(i,j) = atof(s);
            s = strtok(NULL, " ");
        }
    fs.close();
}
void check_data(vector<vector<double>> &x, mat &y, int q){
    for (int i=0; i<784; i++){
        if (i%28==0)    cout << endl;
        cout << (int)x[q][i] << " ";
    }
    for (int i=0; i<392; i++){
        if (i%28==0)    cout << endl;
        cout << (int)x[q][i] << " ";
    }
    for (int i=0; i<392; i++){
        if (i%28==0)    cout << endl;
        if (y(q,i)>0.5)  cout << "1 ";
        else  cout << "0 ";
    }
}
/*
template<typename T>
void test(T &data, vector<Layer> &nn){
    int max_itr = data.size()/batch;
    double llv[max_itr];
    double loss[batch][392];
    memset(llv, 0, sizeof(llv));
    fstream fs;
    fs.open("test_loss.txt", ios::out);

    for (int itr=0; itr<data.size(); itr+=batch){




        for (int i=0; i<batch; i++)
            for (int j=0; j<392; j++)
                llv[itr/batch] += loss[i][j];
        llv[itr/batch] /= batch;
    }
    for (int i=0; i<max_itr; i++)
        fs << i << "\t" << llv[i] << endl;
    fs.close();
}

    Mat img(28, 28, CV_8U);
    for (int i=0; i<28; i++)
        for (int j=0; j<28; j++)
            img2.at<uchar>(i, j) = dataset.test_images[0][j+i*28];
    imshow("222", img2);
*/


#include <iostream>
#include <fstream>
#include <armadillo>
#include <iomanip>
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace arma;
//using namespace cv;

enum _sample{STC, DET, GB};
enum _estimator{LR, ST, aGB};
pair <_sample, _estimator> se(STC, LR);

const int max_epoch = 1;
const int batch = 40;
double alpha = 3e-5;
double _gamma = 0.9;

class Layer{
    public:
        Layer(int m, int n): v(m,n), a(m,n), s(m,n+1), as(m,n+1){}
        void reset(){
            v.zeros(); 
            a.zeros();
            as.zeros();
        }
        mat v, a, s, as;
};
class Weights{
    public:
        Weights(int m, int n): v(n,m), a(n,m), mt(n,m){}
        void init(){
            mt.zeros();
            v.randu();
            v -= 0.5;
        }
        mat v, a, mt;
};
void sample(Layer &h){
    int r = h.v.n_rows - 1, c = h.v.n_cols - 1;
    switch(se.first){
        case STC:
            h.s.submat(0, 0, r, c) = conv_to<mat>::from(mat(r+1, c+1, fill::randu) > h.v);
            break;
        case DET:
            h.s.submat(0, 0, r, c) = h.v;
            break;
        case GB:
            mat p1 = exp(log(h.v)-log(-log(mat(r+1, c+1, fill::randu))));
            mat p0 = exp(log(1.0-h.v)-log(-log(mat(r+1, c+1, fill::randu))));
            h.s.submat(0, 0, r, c) = p1 / (p0 + p1);
            break;
    }
}
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
void back_prop(Layer &y, Weights &w, Layer &h){
    y.a %= y.v % (1.0-y.v);
    w.a += h.s.t() * y.a;
    h.as += y.a * w.v.t();
}
void a_sample(Layer &h, Layer &y){
    int r = h.v.n_rows - 1, c = h.v.n_cols - 1;
    mat hh(r, c);
    switch(se.second){
        case LR:
            hh = conv_to<mat>::from( h.s.submat(0, 0, r, c)>=h.v ) % h.v;
            hh += conv_to<mat>::from( h.s.submat(0, 0, r, c)<h.v) % (1.0-h.v);
            h.a += 1/hh;
            break;        
        case ST:
            h.a += h.as.submat(0, 0, r, c);
            break;
        case aGB:
            h.a += h.s.submat(0, 0, r, c)%(1-h.s.submat(0, 0, r, c)) / h.v % h.as.submat(0, 0, r, c);
            break;
    }
}
void optimizer(Weights &w){
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
            fs << setprecision(17) << fixed << w.v(i,j) << " ";
    fs.close();
}/*
void read_w(MTX &w, int n, int m, string c){
    char buf[n*m*50], *s;
    ifstream fs(c);
    fs.read(buf, sizeof(buf));
    
    s = strtok(buf," ");
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++){
            w[i][j] = atof(s);
            s = strtok(NULL," ");
        }
    fs.close();
}
void check_data(MTX &x, MTX &y, int q){
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
        if (y[q][i]>0.5)  cout << "1 ";
        else  cout << "0 ";
    }
}
void check_gd(int lev, int pi, int pj, vector<Layer> &nn, vector<S_Layer> &ss, vector<Weights> &ww,
                  mat &x, std::vector<std::vector<uint8_t>> &data, mat &loss){
    cout << ww[lev].a(pj,pi) << endl;;
    double hh=1e-6;
    vec yy(392);    
    for (int i=0; i<392; i++)    
        yy(i)=loss(0,i);

    for (int i=0; i<nn.size(); i++)
        nn[i].reset();
    for (int i=0; i<ss.size(); i++)
        ss[i].reset();
    for (int i=0; i<batch; i++)
        for (int j=0; j<392; j++)
            loss(i,j) = (double)data[i][j+392];

    ww[lev].v(pj,pi) -= hh;
    feed_forward(x, ww[0], nn[0]);
    for (int i=0; i<nn.size()-1; i++){
        sample(nn[i], ss[2*i]);
        feed_forward(ss[2*i], ww[i+1], ss[2*i+1]);
        expect(ss[2*i+1], nn[i+1]);
    }
    loss_function(nn.back(), loss);

    if (lev==ww.size()-1)
        cout << (yy(pi)-loss(0,pi)) / hh;
    else{
        for (int i=0; i<392; i++)
            yy(i) -= loss(0,i);
        for (int i=1; i<392; i++) 
            yy(0) += yy(i);
        cout << yy(0)/hh << endl;
    }
    char zz;    cin >> zz;
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

    Mat img(28, 28, CV_8U);
    for (int i=0; i<28; i++)
        for (int j=0; j<28; j++)
            img2.at<uchar>(i,j) = dataset.test_images[0][j+i*28];
    imshow("222",img2);
*/

int main(){
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    binarize_dataset(dataset);
  // srand(time(NULL));

    const int v_d = 392;
    const int h_d = 200;
    mat x(batch, v_d+1);
    x.col(v_d).ones();
    mat loss(batch, v_d);

    vector<Layer> nn;
    nn.push_back(Layer(batch, h_d));                 //    h1
    nn.back().s.col(h_d).ones();
    nn.push_back(Layer(batch, h_d));                 //    h2
    nn.back().s.col(h_d).ones();
    nn.push_back(Layer(batch, v_d));                 //    y

    vector<Weights> ww;
    ww.push_back(Weights(h_d, v_d+1));              //    wxh1
    ww.back().init();
    ww.push_back(Weights(h_d, h_d+1));              //    wh1h2
    ww.back().init();
    ww.push_back(Weights(v_d, h_d+1));              //    wh2y
    ww.back().init();

    int max_itr = dataset.training_images.size()/batch;
    mat gd(3, max_itr);
    vec ll(max_itr);
    ofstream fs("train_loss.txt");

    for (int epoch=0; epoch<max_epoch; epoch++){
    //    shuffle(begin(dataset.training_images), end(dataset.training_images), default_random_engine(time(NULL)));
        ll.zeros();
        for (int itr=0; itr<dataset.training_images.size(); itr+=batch){
            for (int i=0; i<nn.size(); i++)
                nn[i].reset();
            for (int i=0; i<ww.size(); i++)
                ww[i].a.zeros();
            for (int i=0; i<batch; i++)
                for (int j=0; j<v_d; j++){
                    x(i,j) = (double)dataset.training_images[i+itr][j];
                    loss(i,j) = (double)dataset.training_images[i+itr][j+v_d];
                }
            nn.back().a.ones();

            feed_forward(x, ww[0], nn[0]);
            for (int i=0; i<nn.size()-1; i++){
                sample(nn[i]);
                feed_forward(nn[i].s, ww[i+1], nn[i+1]);
            }
            loss_function(nn.back(), loss);

            for (int i=nn.size()-1; i>0; i--){
                back_prop(nn[i], ww[i], nn[i-1]); 
                a_sample(nn[i-1], nn[i]);
            }
            back_prop_w(nn[0], ww[0], x);
    //        check_gd(1,2,22, nn, ss, ww, x, dataset.training_images,loss);
            for (int i=0; i<ww.size(); i++){
                optimizer(ww[i]);
                gd(i,itr/batch) = norm(ww[i].a, 2);
            }
            ll(itr/batch) = accu(loss)/batch;
        }
        for (int i=0; i<max_itr; i++)
            fs << i+epoch*max_itr << "\t" << ll(i) << "\t" << gd(2,i) <<"\t" << gd(1,i) << "\t" << gd(0,i) << endl;
        cout << "epoch: " << epoch << " is done" << endl;
    }
    fs.close();
    for (int i=0; i<ww.size(); i++)
        write_w(ww[i], i);

    return 0;
}


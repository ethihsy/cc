#include <iostream>
#include <fstream>
#include <armadillo>
#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"
#include "nn.hpp"
#include "gd.hpp"

using namespace std;
using namespace arma;

const int max_epoch = 1;
const int batch = 40;

enum _sample{STC, DET, GB, uPF};
enum _estimator{LR, ST, aGB, uPB};
pair <_sample, _estimator> se(STC, LR);

int main(){
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    binarize_dataset(dataset);
 //   srand(time(NULL));

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
    ww.push_back(Weights(h_d, v_d));                 //    wxh1
    ww.back().init();
    ww.push_back(Weights(h_d, h_d));                 //    wh1h2
    ww.back().init();
    ww.push_back(Weights(v_d, h_d));                 //    wh2y
    ww.back().init();

        vector<Layer> cc;
        cc.push_back(Layer(batch, h_d));
        cc.push_back(Layer(batch, v_d));

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
                    x(i, j) = (double)dataset.training_images[i+itr][j];
                    loss(i, j) = (double)dataset.training_images[i+itr][j+v_d];
                }
            nn.back().a.ones();

            feed_forward(x, ww[0], nn[0]);
            for (int i=0; i<nn.size()-1; i++)
                switch(se.first){
                    case DET:
                        DET_feed_forward(nn[i], ww[i+1], nn[i+1]);
                        break;
                    case GB:
                        GB_feed_forward(nn[i], ww[i+1], nn[i+1]);
                        break;
                    case STC:
                        STC_feed_forward(nn[i], ww[i+1], nn[i+1]);
                        break;
                    case uPF:
                        uP_feed_forward(nn[i], ww[i+1], nn[i+1], cc[i]);
                        break;
                }
            loss_function(nn.back(), loss);

            for (int i=nn.size()-1; i>0; i--){
                back_prop_w(nn[i], ww[i], nn[i-1].s);
                switch(se.second){
                    case ST:
                        ST_back_prop_x(nn[i], ww[i], nn[i-1]);
                        break;
                    case aGB:
                        aGB_back_prop_x(nn[i], ww[i], nn[i-1]);
                        break;
                    case LR:
                        LR_back_prop_x(nn[i], ww[i], nn[i-1], i, loss);
                        break;
                    case uPB:
                        uP_back_prop_x(nn[i], ww[i], nn[i-1], i, loss);
                        break;
                }
            }
            back_prop_w(nn[0], ww[0], x);
    //        check_gd(1, 2, 22, nn, ss, ww, x, dataset.training_images, loss);
            for (int i=0; i<ww.size(); i++){
                optimizer(ww[i], batch);
                gd(i, itr/batch) = norm(ww[i].a, 2);
            }
            ll(itr/batch) = accu(loss)/batch;
        }
        for (int i=0; i<max_itr; i++)
            fs << i+epoch*max_itr << "\t" << ll(i) << "\t" << gd(2, i) <<"\t" << gd(1, i) << "\t" << gd(0, i) << endl;
        cout << "epoch: " << epoch << " is done" << endl;
    }
    fs.close();
    for (int i=0; i<ww.size(); i++)
        write_w(ww[i], i);

    return 0;
}


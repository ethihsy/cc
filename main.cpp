#include <iostream>
#include <fstream>

#include "mnist/include/mnist/mnist_reader.hpp"
#include "mnist/include/mnist/mnist_utils.hpp"
#include "n.hpp"
#include "u.hpp"

using namespace std;

int main(){
   auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
   binarize_dataset(dataset);
   srand(time(NULL));

   const int v_d = 392;
   const int h_d = 200;
   mat x(batch, v_d+1);
   mat loss(batch, v_d);

   vector<Layer> nn;
   nn.push_back(Layer(batch, h_d));             //   h1
   nn.push_back(Layer(batch, h_d));             //   h2
   nn.push_back(Layer(batch, v_d));             //   y

   vector<S_Layer> ss;
   ss.push_back(S_Layer(batch, h_d+1, n_s));    //   sh1
   ss.push_back(S_Layer(batch, h_d, n_s));      //   sh1y
   ss.push_back(S_Layer(batch, h_d+1, n_s));    //   sh2
   ss.push_back(S_Layer(batch, v_d, n_s));      //   sh2y

   vector<Weights> ww;
   ww.push_back(Weights(h_d, v_d+1));           //   wxh1
   ww.back().init();
   ww.push_back(Weights(h_d, h_d+1));           //   wh1h2
   ww.back().init();
   ww.push_back(Weights(v_d, h_d+1));           //   wh2y
   ww.back().init();
// ------------------------------------------------------------------------------   
   int max_itr = dataset.training_images.size()/batch;
   mat gd(3, max_itr);
   vec ll(max_itr);
   ofstream fs("train_loss.txt");

   for (int epoch=0; epoch<max_epoch; epoch++){
      shuffle(begin(dataset.training_images), end(dataset.training_images), default_random_engine(time(NULL)));
      ll.zeros();
      for (int itr=0; itr<dataset.training_images.size(); itr+=batch){
         for (int i=0; i<nn.size(); i++)
            nn[i].reset();
         for (int i=0; i<ss.size(); i++)
            ss[i].reset();
         for (int i=0; i<ww.size(); i++)
            ww[i].a.zeros();
         for (int i=0; i<batch; i++)
            for (int j=0; j<v_d; j++){
               x(i,j) = (double)dataset.training_images[i+itr][j];
               loss(i,j) = (double)dataset.training_images[i+itr][j+v_d];
            }
         x.col(v_d).ones();
         nn.back().a.ones();

         feed_forward(x, ww[0], nn[0]);
         for (int i=0; i<nn.size()-1; i++){
            sample(nn[i], ss[2*i]);
            feed_forward(ss[2*i], ww[i+1], ss[2*i+1]);
            expect(ss[2*i+1], nn[i+1]);
         }
         loss_function(nn.back(), loss);

         for (int i=nn.size()-1; i>0; i--){
            back_prop(nn[i], ss[2*i-1], ww[i], ss[2*i-2]); 
            a_sample(ss[2*i-2], nn[i-1], ss[2*i-1], nn[i]);
         }
         back_prop_w(nn[0], ww[0], x);
   //      check_gd(1,2,22, nn, ss, ww, x, dataset.training_images,loss);
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

//   test(dataset.test_images);

   return 0;
}


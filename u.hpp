#include <fstream>

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
   for (int i=0; i<w._v.size(); i++)
      for (int j=0; j<w._v.front().size(); j++)
         fs << setprecision(17) << fixed << w._v[i][j] << " ";
   fs.close();
}
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
      if (i%28==0)   cout << endl;
      cout << (int)x[q][i] << " ";
   }
   for (int i=0; i<392; i++){
      if (i%28==0)   cout << endl;
      cout << (int)x[q][i] << " ";
   }
   for (int i=0; i<392; i++){
      if (i%28==0)   cout << endl;
      if (y[q][i]>0.5)  cout << "1 ";
      else  cout << "0 ";
   }
}
void check_gd(int lev, int pi, int pj, vector<Layer> &nn, vector<Weights> &ww, MTX &x, MTX &data, MTX &loss){
   cout << ww[lev]._a[pi][pj] << endl;;
   double hh=1e-6;
   double yy[v_d];   
   for (int i=0; i<v_d; i++)   
      yy[i]=loss[0][i];

   for (int i=0; i<nn.size(); i++)
      nn[i].reset();
   nn.back().seed();
   for (int i=0; i<ww.size(); i++)
      ww[i].reset();
   for (int i=0; i<batch; i++){
      x[i][v_d] = 1;
      for (int j=0; j<v_d; j++)
         loss[i][j] = (double)data[i][j+v_d];
   }
   ww[lev]._v[pi][pj] -= hh;
   feed_forward(x, ww[0], nn[0]);
   for (int i=0; i<5; i+=3){
      sample(nn[i], nn[i+1]);
      feed_forward(nn[i+1]._v, ww[(i+3)/3], nn[i+2], n_s);
      expect(nn[i+2], nn[i+3], n_s);
   }
   loss_function(nn.back(), loss);

   if (lev==ww.size()-1)
      cout << (yy[pi]-loss[0][pi]) / hh;
   else{
      for (int i=0; i<v_d; i++)
         yy[i]-=loss[0][i];
      for (int i=1; i<v_d; i++) 
         yy[0] += yy[i];
      cout << yy[0]/hh << endl;
   }
   char zz;   cin >> zz;
}/*
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

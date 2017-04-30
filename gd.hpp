#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

void feed_forward(Layer &h, Weights &w, Layer &y){
    y.v = h.s * w.v;
    y.v = 1.0 / (1.0+exp(-y.v));
}
//  -------------------------------------------------------------------------------
void DET_feed_forward(Layer &h, Weights &w, Layer &y){
    h.s.submat(0, 0, h.v.n_rows-1, h.v.n_cols-1) = h.v;
    feed_forward(h, w, y);
}
void STC_feed_forward(Layer &h, Weights &w, Layer &y){
    int r = h.v.n_rows - 1;
    int c = h.v.n_cols - 1;
    h.s.submat(0, 0, r, c) = conv_to<mat>::from(mat(r+1, c+1, fill::randu) > h.v);
    feed_forward(h, w, y);
}
//  -------------------------------------------------------------------------------
void ST_back_prop_x(Layer &y, Weights &w, Layer &h){
    h.as += y.a * w.v.t();
    h.a += h.as.submat(0, 0, h.v.n_rows-1, h.v.n_cols-1);
}
//  -------------------------------------------------------------------------------
void GB_feed_forward(Layer &h, Weights &w, Layer &y){
    int r = h.v.n_rows - 1;
    int c = h.v.n_cols - 1;

    mat p1 = exp(log(h.v)-log(-log(mat(r+1, c+1, fill::randu))));
    mat p0 = exp(log(1.0-h.v)-log(-log(mat(r+1, c+1, fill::randu))));
    h.s.submat(0, 0, r, c) = p1 / (p0 + p1);
    
    feed_forward(h, w, y);
}
void aGB_back_prop_x(Layer &y, Weights &w, Layer &h){
    int r = h.v.n_rows - 1;
    int c = h.v.n_cols - 1;

    h.as += y.a * w.v.t();
    h.a += h.s.submat(0, 0, r, c)%(1.0-h.s.submat(0, 0, r, c)) / h.v % h.as.submat(0, 0, r, c);
}
//  -------------------------------------------------------------------------------
void LR_back_prop_x(Layer &y, Weights &w, Layer &h, int level, mat &loss){
    int r = h.v.n_rows - 1;
    int c = h.v.n_cols - 1;
    vec acl;

    mat hh = conv_to<mat>::from(h.s.submat(0, 0, r, c)>=h.v) % h.v;
    hh += conv_to<mat>::from(h.s.submat(0, 0, r, c)<h.v) % (1.0-h.v);
    hh = clamp(hh, 0.05, 1);

    if (level<2){
        hh /= y.a;
        acl = sum(y.v, 1);
    }
    else
        acl = sum(loss, 1);

    for (int i=0; i<r+1; i++)
        h.a.row(i) += acl(i)/hh.row(i);
}
//  -------------------------------------------------------------------------------
void uP_feed_forward(Layer &h, Weights &w, Layer &y, Layer &c){
    DET_feed_forward(h, w, c);
    
    STC_feed_forward(h, w, y);
    
}
void uP_back_prop_x(Layer &y, Weights &w, Layer &h, int level, mat &loss){
    int r = h.v.n_rows - 1;
    int c = h.v.n_cols - 1;
    vec acl;

    mat hh = conv_to<mat>::from(h.s.submat(0, 0, r, c)>=h.v) % h.v;
    hh += conv_to<mat>::from(h.s.submat(0, 0, r, c)<h.v) % (1.0-h.v);
    hh = clamp(hh, 0.1, 1);

    if (level<2){
        hh /= y.a;
        acl = sum(y.v, 1);
    }
    else
        acl = sum(loss, 1);
    for (int i=0; i<r+1; i++)
        h.a.row(i) += acl(i)/hh.row(i);
}


#ifndef MATREAD_H
#define MATREAD_H
#include<mat.h>
#include<iostream>
// #include<opencv2/opencv.hpp>
#include<complex>
#include<string>
using namespace std;
// using namespace cv;
complex<float>*  matToArray(const char *filename, const char *variable); 
#endif

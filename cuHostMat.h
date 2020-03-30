#ifndef CUHOSTMAT_H
#define CUHOSTMAT_H
#include<iostream>
#include<cuda_runtime.h>
#include<cuComplex.h>
using namespace std;
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))

class cuHostMat
{
    // 在CPU中显示声明的mat类
    private:
        int height;                  
        int width;                    
        size_t pitch;                                                   // align to speed up memory access see cudaMallocPitch
        cuComplex *data;
    public:
    __host__ cuHostMat(int h, int w);
    __host__ ~cuHostMat();
    __host__ __device__ int get_h() const;
    __host__ __device__ int get_w() const;
    __host__ __device__ size_t get_pitch() const;
    __host__ __device__ cuComplex * get_data() const;
    __host__ __device__ void init(int h, int w);
    __host__ void set_val(int h, int w, cuComplex *src);
    __device__ cuComplex at(int i, int j) const;                         // access elements in the device
    __device__ void set(int i, int j, cuComplex x);
    __device__  void display() const;                                    // device print data using in kernel
    __host__ void get_val(cuComplex *dst) const;
    __host__ void pad(cuHostMat &res, const int &pad_row, const int &pad_col);
    friend ostream& operator<<(ostream &out, const cuHostMat &mat);
};
#endif
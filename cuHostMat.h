#ifndef CUHOSTMAT_H
#define CUHOSTMAT_H
#include<iostream>
#include<cuda_runtime.h>
#include<cuComplex.h>
using namespace std;
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))
#define BDIMX 32
#define BDIMY BDIMX


class cuHostMat
{
    private:
        int height;                  
        int width;                    
        size_t pitch;                                                   // align to speed up memory access see cudaMallocPitch
        cuComplex *data;
    public:
    __host__ cuHostMat(int h, int w);
    __host__ cuHostMat(const cuHostMat &mat);
    __host__ ~cuHostMat();
    __host__ void release();                                             // explicit call to free GPU memory space
    __host__ __device__ int get_h() const;
    __host__ __device__ int get_w() const;
    __host__ __device__ size_t get_pitch() const;
    __host__ __device__ cuComplex * get_data() const;
    __host__ void setVal(cuComplex *src, int h, int w);                   // copy data from CPU to GPU
    __host__ void getVal(cuComplex *dst) const;                           // copy data from GPU to CPU
    __host__ cuHostMat pad(const int &pad_row, const int &pad_col);   // pad matrix 
    __host__ cuHostMat transpose();
    __device__ cuComplex & at(int i, int j) const;                        // access elements in the device return reference type
    __device__ void set(int i, int j, cuComplex x);
    __device__  void display();                                           // device print data using in kernel
    friend ostream& operator<<(ostream &out, const cuHostMat &mat);       // cout use in host
};
#endif
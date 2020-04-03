#ifndef CUDEVMAT_H_
#define CUDEVMAT_H_
#include<cuda_runtime.h>
#include<stdio.h>
#include<cuComplex.h>
#include<cublas_v2.h>
#include"cuMacro.h"
// write devmat because when in kernel dynamicly allocate 1kB space and the number of thread is 32x32 it will throw "out of memory" exception
// so we allocate the memory in host thread and set a pointer to record the kernel space usage
class cuDevMat{
    private:
        int height;
        int width;
        cuComplex *data;
        char *begin;                                       // the begin position of the thread
        size_t *pointer;                                   // use shared memory to record every thread space usage 
    public:
    __device__ cuDevMat();
    __device__ cuDevMat(int h, int w, char *threadMem, size_t *p); 
    __device__ cuDevMat(const cuDevMat &mat);
    __device__ ~cuDevMat();
    __device__ void init(int h, int w, char *threadMem, size_t *p);
    __device__ void release();
    __device__ int get_h() const;
    __device__ int get_w() const;
    __device__ cuComplex * get_data() const;
    __device__ char * get_mem() const;
    __device__ size_t * get_pointer() const;
    __device__ cuComplex & at(int i, int j) const;                        // access elements in the device return reference type
    __device__ void set(int i, int j, cuComplex x);
    __device__ void display() const;
    __device__ cuComplex det();
    __device__ cuDevMat inv();                                           // only for 3X3
    __device__ cuDevMat invParal();
    __device__ cuDevMat mul(cuDevMat mat, cuComplex alpha);
    __device__ cuDevMat her(cuComplex alpha);
};
#endif
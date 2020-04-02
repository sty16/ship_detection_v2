#include"cuHostMat.h"
#include<stdio.h>



__global__ void test()        //kernel 不能传指针或者引用 cuHostMat:变量height等在CPU内存上，指针指向的数据块在GPU显存上
{
    double *temp;
    if(threadIdx.x <= 80 && threadIdx.y <= 80)
    {
        int N = 830;
        cudaError_t error_t = cudaMalloc((void**)&temp, sizeof(double)*N);
        if(error_t != cudaSuccess)
        {
            printf("%s, %d, %d\n", cudaGetErrorString(error_t), threadIdx.x, threadIdx.y);
        }else{
            for(int i=0;i<N;i++)
            {
                temp[i] = (double) i;
            }
        }

    }
    __syncthreads();
    if(threadIdx.x == 2&&threadIdx.y == 2)
    {
        printf("%f\n", temp[2]);
    }
    cudaFree(temp);
    
}

int main()
{
    int N = 6, pad_row=2, pad_col=2;
    cuHostMat temp(6, 6);
    cuComplex *a = new cuComplex[N*N], *b = new cuComplex[36];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            a[INDEX(i, j, N)] = make_cuComplex(i, j);
        }
    }
    int ngpus, canAccessPeer;
    cudaGetDeviceCount(&ngpus);
    cudaError_t error_t= cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
    printf("%d, %d\n", ngpus, canAccessPeer);
    temp.setVal(a, N, N);                                                 //对象以值的传递方式传入函数形参需要调用拷贝构造函数
    cuHostMat temp_pad = temp.pad(pad_row, pad_col);
    cuHostMat temp_t = temp.transpose();
    cout<<temp<<endl;
    cout<<temp_pad<<endl;
    cout<<temp_t<<endl;
    temp.release();
    temp_pad.release();
    temp_t.release();
    dim3 blockdim(32,32);
    dim3 griddim(1, 1);
    // cudaSetDevice(0);
    // test<<<griddim, blockdim>>>();
    // cudaDeviceSynchronize();
}
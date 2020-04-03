#include"cuHostMat.h"
#include"cuDevMat.h"
#include"matRead.h"
#include<stdio.h>
__device__ __managed__ char *MemPool;            // the global memory that thread have  内存池


__global__ void test()        //kernel 不能传指针或者引用 cuHostMat:变量height等在CPU内存上，指针指向的数据块在GPU显存上
{
    __shared__ size_t pointer[2048];
    // size_t pointer = 0;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int num = i*blockDim.y + j;
    // double *temp;
    // if(threadIdx.x <= 80 && threadIdx.y <= 80)
    // {
    //     int N = 830;
    //     cudaError_t error_t = cudaMalloc((void**)&temp, sizeof(double)*N);
    //     if(error_t != cudaSuccess)
    //     {
    //         printf("%s, %d, %d\n", cudaGetErrorString(error_t), threadIdx.x, threadIdx.y);
    //     }else{
    //         for(int i=0;i<N;i++)
    //         {
    //             temp[i] = (double) i;
    //         }
    //     }

    // }
    // __syncthreads();
    // if(threadIdx.x == 2&&threadIdx.y == 2)
    // {
    //     printf("%f\n", temp[2]);
    // }
    // cudaFree(temp);
    char *ThreadMemPool = (char *)MemPool + (size_t)num*THREADSPACE;   // 找到线程自己的全局内存池位置
    cuDevMat a(3, 3, ThreadMemPool, &pointer[num]);
    for(int i=0;i<3;i++){
        for(int j = 0;j<3;j++){
            a.set(i, j, make_cuComplex(i*3+j+1,0));
        }
    }
    a.set(0, 2, make_cuComplex(1, 0));
    // a.set(1, 2, make_cuComplex(0, 0));
    // a.set(2, 1, make_cuComplex(1, 0));
    cuComplex det = a.det();
    cuDevMat a_her = a.her(make_cuComplex(1, 0));
    __syncthreads();
    if(i==0 && j==0)
    {
        printf("%f,%f\n",det.x, det.y);
        a_her.display();
        // printf("%f\n", a_inv.at(0, 0).x);
    }

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
    dim3 blockdim(32, 32);
    dim3 griddim(1, 1);
    size_t size = (size_t)THREADSPACE*blockdim.x*blockdim.y*griddim.x*griddim.y;
    printf("%lu", size);
    error_t = cudaMalloc((void **)&MemPool, sizeof(char)*size); //分配的不是全局内存空间
    switch( error_t )
    {
      case cudaSuccess: printf("cudaSuccess\n");break;
      case cudaErrorMemoryAllocation:printf("cudaErrorMemoryAllocation\n");break;
      default:printf("default: %d \n",error_t );break;
    }
    // cudaSetDevice(0);
    test<<<griddim, blockdim>>>();
    cudaDeviceSynchronize();
}
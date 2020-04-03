#include"cuHostMat.h"
#include"cuDevMat.h"
#include"matRead.h"
#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<sys/time.h>
using namespace cv;

__device__ __managed__ cuHostMat *imgPad[NGPUS];          //device variable managed variable,to which Host is accessible
__device__ __managed__ cuHostMat *img[NGPUS];             // the matrix of the origin image
__device__ __managed__ char *MemPool[NGPUS];            // the global memory that thread have  内存池
__device__ __managed__ uint8 *resImg[NGPUS];             // the  result of the pwf image

inline void enableP2P(int ngpus)
{
    for(int i = 0;i < ngpus; i++) 
    {
        cudaSetDevice(i);
        for(int j = 0;j < ngpus; j++)
        {
            if(i == j)
            {
                continue;
            }
            int peer_access_available = 0;
            cudaDeviceCanAccessPeer(&peer_access_available, i, j);
            if(peer_access_available)
            {
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

__device__ void copyToClutter(cuComplex *clutter_data, int height, int width, int row, int col, int pad_row, int pad_col, int device)
{
    // pad_row = 7 pad_col = 7
    for(int k=0;k<height;k++)
    {
        for(int i=0;i<2*pad_row+1;i++)
        {
            for(int j = 0;j<2*pad_col+1;j++)
            {
                int temp = i*(2*pad_col+1) + j;
                if(i==(row+pad_row) && j==(col+pad_col))
                {
                    continue;
                }
                clutter_data[INDEX(k, temp, width)] = imgPad[device][k].at(row+i, col+j);
            }
        }
    }
}

__host__ float  get_T_PWF(int num, float P, int maxIter=500, double ep = 1e-5, double x0 = 1)
{
    double x, fx, dfx;
    double Pfa = (double)P;
    for (int i=0; i< maxIter;i++)
    {
        fx = exp(-x0) * (1 + x0 + pow(x0, 2)/ 2) - Pfa;
        dfx = -exp(-x0) * (pow(x0, 2)/ 2);
        x = x0 - fx / dfx;
        if ((fx < ep) and (fx > -ep)) break; 
        x0 = x;
    }
    return (float)x;
}

__global__ void f_PWF(int device, int height, int width, int pad_row, int pad_col, int num, float T) 
{
    __shared__ size_t pointer[1024];
    int c_row, t_row, t_col, N=(2*pad_row+1)*(2*pad_col+1) - 1;                // set the pointer 0 && set the sliding window size 30
    cuComplex  data[3], temp[3], result;
    float res;
    c_row = blockIdx.x*gridDim.y + blockIdx.y;
    t_row = c_row + num*gridDim.x*gridDim.y;
    t_col = INDEX(threadIdx.x, threadIdx.y, blockDim.y);
    pointer[t_col] = 0;
    if(t_row < height && t_col < width)
    {
        char *ThreadMemPool = (char *)MemPool[device] + (size_t)INDEX(c_row, t_col, width)*THREADSPACE;   // 找到线程自己的全局内存池位置
        cuDevMat clutter(3, N, ThreadMemPool, &pointer[t_col]);
        copyToClutter(clutter.get_data(), 3, N, t_row, t_col, pad_row, pad_col, device);
        cuComplex alpha = make_cuComplex(1.0/N, 0);
        cuDevMat sigma_c = clutter.her(alpha);
        cuDevMat sigma_inv = sigma_c.inv();
        for(int i=0;i < 3;i++)
        {
            data[i] =  img[device][i].at(t_row, t_col);                    // 注意mallocpitch要行对其访问
        }
        for(int i=0;i<3;i++)
        {
            temp[i] = make_cuComplex(0, 0);
            for(int j=0;j<3;j++)
            {
                temp[i] = cuCaddf(temp[i], cuCmulf(sigma_inv.at(i, j), data[j]));
            }
        }
        result = make_cuComplex(0, 0);
        for(int i=0;i<3;i++)
        {
            result = cuCaddf(result, cuCmulf(cuConjf(data[i]), temp[i]));
        }
        res = cuCabsf(result);
        resImg[device][INDEX(t_row, t_col, width)] = res>T?255:0;
    }
}

int main(){
    const char *matfile_HH = "./data/imagery_HH.mat";
    const char *param_HH = "imagery_HH";
    const char *matfile_HV = "./data/imagery_HV.mat";
    const char *param_HV = "imagery_HV";
    const char *matfile_VV = "./data/imagery_VV.mat";
    const char *param_VV = "imagery_VV";
    complex<float> *img_mat[3];
    int h = 1000, w = 1000, N = 7;                               //size of the image data
    float Pfa = 0.001;
    uint8 *resImg_host;
    cudaStream_t stream[NGPUS];
    cudaError_t error_t;
    float T = get_T_PWF(3, Pfa);
    img_mat[0] = matToArray(matfile_HH, param_HH);
    img_mat[1] = matToArray(matfile_HV, param_HV);
    img_mat[2] = matToArray(matfile_VV, param_VV);
    dim3 griddim(16,8);
    dim3 blockdim(32,32);
    size_t size = (size_t)THREADSPACE*blockdim.x*blockdim.y*griddim.x*griddim.y;
    int ngpus;
    cudaGetDeviceCount(&ngpus);
    enableP2P(ngpus);
    for(int i=0;i<ngpus;i++)
    {
        cudaSetDevice(i);     
        cudaStreamCreate(&stream[i]);
        cudaMallocManaged((void**)&img[i], sizeof(cuHostMat)*3);
        error_t = cudaMalloc((void **)&resImg[i], sizeof(uint8)*h*w);
        switch( error_t )
        {
        case cudaSuccess: break;
        case cudaErrorMemoryAllocation:printf("cudaErrorMemoryAllocation\n");break;
        default:printf("default: %d \n",error_t );break;
        }
        error_t = cudaMalloc((void **)&MemPool[i], sizeof(char)*size); 
        switch( error_t )
        {
        case cudaSuccess: break;
        case cudaErrorMemoryAllocation:printf("cudaErrorMemoryAllocation\n");break;
        default:printf("default: %d \n",error_t );break;
        }
        if(i == 0)
        {
            cudaMallocManaged((void **)&imgPad[i], sizeof(cuHostMat)*3);
            for(int j=0;j<3;j++)
            {
                img[i][j].init(h, w);
                img[i][j].setVal((cuComplex *)img_mat[j], h, w);  //bug 1 i mistake for j
                imgPad[i][j] = img[i][j].pad(N, N);    // pad to use sliding windows    
            }
        }else{
            cudaMallocManaged((void **)&imgPad[i], sizeof(cuHostMat)*3);
            for(int j=0;j<3;j++)
            {
                img[i][j].init(h, w);
                imgPad[i][j].init(h+2*N, w+2*N);
                error_t = cudaMemcpy2D(imgPad[i][j].get_data(), imgPad[i][j].get_pitch(), imgPad[0][j].get_data(), imgPad[0][j].get_pitch(), sizeof(cuComplex)*(w+2*N), h+2*N, cudaMemcpyDeviceToDevice);
                switch( error_t )
                {
                case cudaSuccess: break;
                case cudaErrorMemoryAllocation:printf("cudaErrorMemoryAllocation\n");break;
                default:printf("default: %d \n",error_t );break;
                }
                error_t = cudaMemcpy2D(img[i][j].get_data(), img[i][j].get_pitch(), img[0][j].get_data(), img[0][j].get_pitch(), sizeof(cuComplex)*(w), h, cudaMemcpyDeviceToDevice);
                switch( error_t )
                {
                case cudaSuccess: break;
                case cudaErrorMemoryAllocation:printf("cudaErrorMemoryAllocation\n");break;
                default:printf("default: %d \n",error_t );break;
                }
            }
        }
    }
    resImg_host = new uint8[h*w];
    double iStart = cpuSecond();
    printf("program begin \n");
    int num = h/(griddim.x*griddim.y*ngpus) + 1;    //每个GPU的kernel数量
    for(int i = 0;i < ngpus; i++)
    {
        cudaSetDevice(i);
        for(int j = 0; j < num; j++)
        {
            f_PWF<<<griddim, blockdim, 0, stream[i]>>>(i, h, w, N, N, j + i*num, T);
        }
    }
    cudaDeviceSynchronize();
    printf("\nfinish waiting for display the result\n");
    double iElaps = cpuSecond() - iStart;
    printf("time usage: %lf\n", iElaps);
    cudaMemcpy(&resImg[0][512000], &resImg[1][512000], sizeof(uint8)*488000, cudaMemcpyDeviceToDevice);
    Mat detect_res = Mat::zeros(h, w, CV_8UC1);
    cudaMemcpy(resImg_host, resImg[0], sizeof(char)*h*w, cudaMemcpyDeviceToHost);
    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            detect_res.at<uchar>(i,j) = resImg_host[INDEX(i, j, w)];
        }
    }
    imshow("detected" , detect_res);
    while(char(waitKey())!='q') 
	{    
    }

    printf("CUDA-capable devices:%d\n", ngpus);
}
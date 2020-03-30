#include"cuHostMat.h"
#include<iostream>


__global__ void cuHostMatPad(cuHostMat orig, cuHostMat res, const int &pad_row, const int &pad_col) //不能像核函数传递指针类型
{
    //  __global__ function(kernel) can not be defined as a class member function
    int i = threadIdx.x + blockDim.x * blockIdx.x;  // the ith row
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i < res.get_h() && j < res.get_w())
    {
        res[i*10+j] = make_cuComplex(1,1);
        printf("ok\n");
        printf("%f\n", res[i*10+j].x);
        // res.set(i, j, make_cuComplex(1, 1));
        // printf("%f", orig.at(1,1).x);
        // if(i<pad_row)
        // {
        //     if(j<pad_col)
        //     {
        //         res.set(i, j, orig.at(pad_row - 1 - i, pad_col - 1 -j));
        //     }
        //     else if(j >= pad_col && j < pad_col + orig.get_w())
        //     {
        //         res.set(i, j, orig.at(pad_row - 1 - i, j - pad_col));
        //     }
        //     else{
        //         res.set(i, j, orig.at(pad_row - 1 - i, 2*orig.get_w() + pad_col - 1 - j));
        //     }
        // }else if(i < pad_row + orig.get_h() && i >= pad_row)
        // {
        //     if(j < pad_col)
        //     {
        //         res.set(i, j, orig.at(i-pad_row, pad_col - 1 -j));
        //     }
        //     else if(j >= pad_col && j < pad_col + orig.get_w())
        //     {
        //         res.set(i, j, orig.at(i-pad_row, j-pad_col));
        //     }
        //     else{
        //         res.set(i, j, orig.at(i-pad_row, 2*orig.get_w() + pad_col - 1 - j));
        //     }
        // }else{
        //     if(j < pad_col)
        //     {
        //         res.set(i, j, orig.at(2*orig.get_h() + pad_row - 1 - i, pad_col - 1 -j));
        //     }
        //     else if(j >= pad_col && j < pad_col + orig.get_w())
        //     {
        //         res.set(i, j, orig.at(2*orig.get_h() + pad_row - 1 - i, j - pad_col));
        //     }
        //     else{
        //         res.set(i, j, orig.at(2*orig.get_h() + pad_row - 1 - i, 2*orig.get_w() + pad_col - 1 - j));
        //     }
        // }
        // printf("(%f, %f)\n", res.at(i, j).x, res.at(i, j).y);
        // printf("%f, %f\n", orig.at(i, j).x, orig.at(i, j).y);
        // printf("\n");
    }
}

__host__ cuHostMat::cuHostMat(int h, int w)
{
    if(h <= 0 || w <= 0)
    {
        fprintf(stderr, "The input size is invalid\n");
        exit(EXIT_FAILURE);
    }else{
        cudaError_t error_t = cudaMallocPitch((void**)&data, &pitch, sizeof(cuComplex)*w, h); 
        if(error_t != cudaSuccess)
        {
            height = width = 0;
            fprintf(stderr, "%s.\n", cudaGetErrorString(error_t));
            exit(EXIT_FAILURE);
        }else{
            height = h;
            width = w;
        }
    }
}

__host__ cuHostMat::~cuHostMat()
{
    cudaFree(data);
}

__host__ __device__ int cuHostMat::get_h() const
{
    return height;
}

__host__ __device__ int cuHostMat::get_w() const
{
    return width;
}

__host__ __device__ size_t cuHostMat::get_pitch() const
{
    return pitch;
}

__host__ __device__ cuComplex * cuHostMat::get_data() const
{
    return data;
}

__host__ __device__ void init(int h, int w)
{
    if(h <= 0 || w <= 0)
    {
        fprintf(stderr, "The input size is invalid\n");
        exit(EXIT_FAILURE);
    }else{
        cudaError_t error_t = cudaMallocPitch((void**)&data, &pitch, sizeof(cuComplex)*w, h); 
        if(error_t != cudaSuccess)
        {
            height = width = 0;
            fprintf(stderr, "%s.\n", cudaGetErrorString(error_t));
            exit(EXIT_FAILURE);
        }else{
            height = h;
            width = w;
        }
    }   
}

__host__ void cuHostMat::set_val(int h, int w, cuComplex *src)
{
    cudaError_t error_t = cudaMemcpy2D(data, pitch, src, sizeof(cuComplex)*w, sizeof(cuComplex)*w, h, cudaMemcpyHostToDevice);
    if(error_t != cudaSuccess)
    {
        fprintf(stderr,"%s.\n", cudaGetErrorString(error_t));
        exit(EXIT_FAILURE);
    }
}

__device__ cuComplex cuHostMat::at(int i, int j) const
{
    if(i < height && j < width)
    {
        return data[INDEX(i, j, pitch/sizeof(cuComplex))];   // pay attention to that pitch but not width and divided by sizeof(cuComplex)
    }else{
        printf("index is out of range\n");
        return data[0];
    }
}

__device__ void cuHostMat::set(int i, int j, cuComplex x)
{
    if(i < height && j < width)
    {
        data[INDEX(i, j, pitch/sizeof(cuComplex))] = x;   // pay attention to pitch
    }else{
        printf("index is out of range\n");
    }
}

__device__ void cuHostMat::display() const
{
    for(int i = 0;i < height;i++)
    {
        for(int j = 0;j < width;j++)
        {
            printf("(%f, %f)", this->at(i, j).x, this->at(i, j).y);
        }
        printf("\n");
    }
}

__host__ void cuHostMat::get_val(cuComplex *dst) const
{
    cudaError_t error_t = cudaMemcpy2D(dst, sizeof(cuComplex)*width, data, pitch, sizeof(cuComplex)*width, height, cudaMemcpyDeviceToHost);
    if(error_t != cudaSuccess)
    {
        fprintf(stderr, "%s.\n", cudaGetErrorString(error_t));
        exit(EXIT_FAILURE);
    }
}

__host__ cuHostMat cuHostMat::pad(cuHostMat &res, const int &pad_row, const int &pad_col)
{
    res.init(height + 2*pad_row, width + 2*pad_col); // 注意局部变量自动析构释放显存空间, 所以需要new的方式显式创建变量
    dim3 blockdim(32, 32);
    dim3 griddim(res.get_h()/32 + 1, res.get_w()/32 + 1);
    cuHostMatPad<<<griddim, blockdim>>>(*this, res, pad_row, pad_col);
    cudaDeviceSynchronize(); 
}

ostream& operator<<(ostream &out, const cuHostMat &mat)
{
    int height = mat.get_h();
    int width = mat.get_w();
    cuComplex *temp = new cuComplex[height*width];
    // mat.get_val(temp);
    for(int i = 0;i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            out<<"("<<temp[INDEX(i, j, width)].x<<", "<<temp[INDEX(i, j, width)].y<<") ";
        }
        out<<endl;
    }
    delete []temp;
    return out;
}
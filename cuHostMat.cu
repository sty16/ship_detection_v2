#include"cuHostMat.h"

__global__ void cuHostMatPad(cuHostMat orig, cuHostMat res, int pad_row, int pad_col) 
{
    //不能像核函数传递指针或引用类型，指针指向的空间在CPU内存, kernel无法访问
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i < res.get_h() && j < res.get_w())
    {
        if(i<pad_row)
        {
            if(j<pad_col)
            {
                res.set(i, j, orig.at(pad_row - 1 - i, pad_col - 1 -j));
            }
            else if(j >= pad_col && j < pad_col + orig.get_w())
            {
                res.set(i, j, orig.at(pad_row - 1 - i, j - pad_col));
            }
            else{
                res.set(i, j, orig.at(pad_row - 1 - i, 2*orig.get_w() + pad_col - 1 - j));
            }
        }else if(i < pad_row + orig.get_h() && i >= pad_row)
        {
            if(j < pad_col)
            {
                res.set(i, j, orig.at(i-pad_row, pad_col - 1 -j));
            }
            else if(j >= pad_col && j < pad_col + orig.get_w())
            {
                res.set(i, j, orig.at(i-pad_row, j-pad_col));
            }
            else{
                res.set(i, j, orig.at(i-pad_row, 2*orig.get_w() + pad_col - 1 - j));
            }
        }else{
            if(j < pad_col)
            {
                res.set(i, j, orig.at(2*orig.get_h() + pad_row - 1 - i, pad_col - 1 -j));
            }
            else if(j >= pad_col && j < pad_col + orig.get_w())
            {
                res.set(i, j, orig.at(2*orig.get_h() + pad_row - 1 - i, j - pad_col));
            }
            else{
                res.set(i, j, orig.at(2*orig.get_h() + pad_row - 1 - i, 2*orig.get_w() + pad_col - 1 - j));
            }
        }
    }
}

__global__ void transposeSmem(cuHostMat a, cuHostMat res)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    __shared__ cuComplex tile[BDIMX][BDIMY];
    int row, col, trow, tcol;
    int m = a.get_h()/blockDim.x;        // the number of full filled block
    int n = a.get_w()/blockDim.y;
    if(blockIdx.x < m && blockIdx.y < n)
    {                                                                              // full block and non-full block
        tile[threadIdx.x][threadIdx.y] = a.at(i,j); 
        int numx;                       // find the index  
        numx = threadIdx.x*blockDim.y + threadIdx.y;
        trow = numx / blockDim.x;
        tcol = numx % blockDim.x;
        row = trow + blockIdx.y*blockDim.y;
        col = tcol + blockIdx.x*blockDim.x;
    }else{
        row = j;col = i;
    }
   __syncthreads();                                                                  //wait for the tile filled with value;
    if(row<res.get_h() && col<res.get_w()){
        if(blockIdx.x < m && blockIdx.y < n){
            res.set(row, col, cuConjf(tile[tcol][trow]));                                        //coalesced  write
        }else{
            res.set(row, col, cuConjf(a.at(i, j)));
        }
    }
}

__host__ cuHostMat::cuHostMat()
{
    height = width = 0;
    pitch = 0;
    data = NULL;
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

__host__ cuHostMat::cuHostMat(const cuHostMat &mat)
{
    height = mat.get_h();
    width = mat.get_w();
    pitch = mat.get_pitch();
    data = mat.get_data();
}

__host__ cuHostMat::~cuHostMat()
{
    // cudaFree(data);                                           // 避免局部变量的析构释放显存空间，注意拷贝构造函数采用的浅复制
}

__host__ void cuHostMat::init(int h, int w)
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

__host__ void cuHostMat::release()
{
    cudaFree(data);
}

__host__ void cuHostMat::setVal(cuComplex *src, int h, int w)
{
    cudaError_t error_t = cudaMemcpy2D(data, pitch, src, sizeof(cuComplex)*w, sizeof(cuComplex)*w, h, cudaMemcpyHostToDevice);
    if(error_t != cudaSuccess)
    {
        fprintf(stderr,"%s.\n", cudaGetErrorString(error_t));
        exit(EXIT_FAILURE);
    }
}

__device__ cuComplex & cuHostMat::at(int i, int j) const
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

__device__ void cuHostMat::display()
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

__host__ void cuHostMat::getVal(cuComplex *dst) const
{
    cudaError_t error_t = cudaMemcpy2D(dst, sizeof(cuComplex)*width, data, pitch, sizeof(cuComplex)*width, height, cudaMemcpyDeviceToHost);
    if(error_t != cudaSuccess)
    {
        fprintf(stderr, "%s.\n", cudaGetErrorString(error_t));
        printf("error");
        exit(EXIT_FAILURE);
    }
}

__host__ cuHostMat cuHostMat::pad(const int &pad_row, const int &pad_col)
{
    cuHostMat res(height+2*pad_row, width+2*pad_col);
    dim3 blockdim(32, 32);
    dim3 griddim(res.get_h()/32 + 1, res.get_w()/32 + 1);
    cuHostMatPad<<<griddim, blockdim>>>(*this, res, pad_row, pad_col);
    cudaDeviceSynchronize(); 
    return res;
}

__host__ cuHostMat cuHostMat::transpose()
{
    cuHostMat res(width, height);
    dim3 blockdim(32, 32); 
    dim3 griddim(height/32 + 1, width/32 + 1);
    transposeSmem<<<griddim, blockdim>>>(*this, res);
    cudaDeviceSynchronize();
    return res;
}

ostream& operator<<(ostream &out, const cuHostMat &mat)
{
    int height = mat.get_h();
    int width = mat.get_w();
    cuComplex *temp = new cuComplex[height*width];
    mat.getVal(temp);
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
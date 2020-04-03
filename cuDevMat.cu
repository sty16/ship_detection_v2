#include"cuDevMat.h"

__global__ void cuMatInv(cuComplex *data, cuComplex *res_data, int height, int width, cuComplex det, char *begin, int threadsize)
{
    // detsize 每个线程的字节数
    __shared__ size_t pointer[256];
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i < height && j < width)
    {
        char *threadMempool = (char *)begin + INDEX(i, j, width)*threadsize;
        pointer[INDEX(i, j, width)] = 0;
        cuDevMat temp(height - 1, width - 1, threadMempool, &pointer[INDEX(i, j, width)]);
        cuComplex *temp_data = temp.get_data();
        for(int k = 0;k < height-1; k++)
        {
            for(int t = 0; t < width-1; t++)
            {
                int row = k>=i?k+1:k;
                int col = t>=j?t+1:t;
                temp_data[INDEX(k, t, width-1)] = data[INDEX(row, col, width)];
            }
        }
        cuComplex  temp_det = temp.det();
        res_data[INDEX(j, i, width)] = cuCdivf(temp_det, det);
        if((i+j)%2 == 1)
        {
            cuComplex alpha = make_cuComplex(-1, 0);
            res_data[INDEX(j, i, width)] = cuCmulf(res_data[INDEX(j, i, width)], alpha);
        }
    }
}


__device__ cuDevMat::cuDevMat()
{
    height = width = 0;
    data = NULL;
    begin = NULL;
    pointer = NULL;
}

__device__ cuDevMat::cuDevMat(int h, int w, char *threadMem, size_t *p)
{
    if(h <= 0 || w <= 0){
        printf("The input size is invalid\n");
    }else{
        begin = threadMem;
        pointer = p;
        if(*pointer + h*w*sizeof(cuComplex) > THREADSPACE)
        {
            printf("out of thread memory");
            height = width = 0;
            data = NULL;
        }else{
            height = h;
            width = w;
            data = (cuComplex *)((char *)begin + *pointer);
            *pointer = *pointer + (size_t)h*w*sizeof(cuComplex);
        }
    }
}

__device__ void cuDevMat::init(int h, int w, char *threadMem, size_t *p)
{
    if(h <= 0 || w <= 0){
        printf("The input size is invalid\n");
    }else{
        begin = threadMem;
        pointer = p;
        if(*pointer + h*w*sizeof(cuComplex) > THREADSPACE)
        {
            printf("out of thread memory");
            height = width = 0;
            data = NULL;
        }else{
            height = h;
            width = w;
            data = (cuComplex *)((char *)begin + *pointer);
            *pointer = *pointer + (size_t)h*w*sizeof(cuComplex);
        }
    }   
}

__device__ cuDevMat::cuDevMat(const cuDevMat &mat)
{
    height = mat.get_h();
    width = mat.get_w();
    data = mat.get_data();
    begin = mat.get_mem();
    pointer = mat.get_pointer();
}

__device__ cuDevMat::~cuDevMat()
{
    // *pointer = *pointer - height*width*sizeof(cuComplex);
    // cuComplex *temp = (cuComplex *)((char *)begin + *pointer);
    // if(temp != data)
    // {
    //     printf("cudaFreeFailure\n");
    //     pointer = pointer + height*width*sizeof(cuComplex);
    // }
}

__device__ void cuDevMat::release()
{
    *pointer = *pointer - height*width*sizeof(cuComplex);
    cuComplex *temp = (cuComplex *)((char *)begin + *pointer);
    if(temp != data)
    {
        printf("cudaFreeFailure\n");
        pointer = pointer + height*width*sizeof(cuComplex);
    }
}

__device__ int cuDevMat::get_h() const
{
    return height;
}

__device__ int cuDevMat::get_w() const
{
    return width;
}

__device__ cuComplex * cuDevMat::get_data() const
{
    return data;
}

__device__ char * cuDevMat::get_mem() const
{
    return (char *)begin;
}

__device__ size_t * cuDevMat::get_pointer() const
{
    return pointer;
}

__device__ cuComplex & cuDevMat::at(int i, int j) const
{
    if(i < height && j < width)
    {
        return data[i*width + j];  
    }else{
        printf("index is out of range\n");
        return data[0];
    }
}

__device__ void cuDevMat::set(int i, int j, cuComplex x)
{
   if(i < height && j < width)
    {
        data[i*width + j] = x;   // pay attention to pitch
    }else{
        printf("index is out of range\n");
    }
}

__device__ void cuDevMat::display() const
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

__device__ cuComplex cuDevMat::det()
{
    if(height != width)
    {
       printf("the height and width of the matrix are not match\n");
       return make_cuComplex(0, 0);
    }
    cuComplex det = make_cuComplex(0, 0);
    if(height == 1)
    {
        det = data[0];
        return det;
    }
    if(height == 2)
    {
        det = cuCsubf(cuCmulf(data[0], data[3]), cuCmulf(data[1], data[2]));
        return det;
    }
    cuDevMat temp(height - 1, width - 1, begin, pointer);
    int row, col;
    cuComplex *temp_data = temp.get_data();
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height - 1; j++)
        {
            for(int k = 0; k < width - 1; k++)
            {
                row = j + 1;
                col = (k>=i)?k+1:k;
                temp_data[INDEX(j, k, width - 1)] = data[row*width + col];
            }
        }
        cuComplex cofactor = temp.det();
        if(i%2 == 0)
        {
            det = cuCaddf(det, cuCmulf(data[i], cofactor));
        }else{
            det = cuCsubf(det, cuCmulf(data[i], cofactor));
        }
    }
    temp.release();
    return det;
}

__device__ cuDevMat cuDevMat::inv()
{
    cuDevMat res;
    if(height != width)
    {
        printf("The matrix is not a square matrix\n");
        return res;
    }else{
        cuComplex det = this->det();
        if(cuCabsf(det) < (float)1e-5)
        {
            printf("the matrix is strange");          //矩阵奇异
            return res;
        }
        res.init(height, width, begin, pointer);
        if(height == 1)
        {
            res.set(0, 0, cuCdivf(make_cuComplex(1, 0), det));
            return res;
        }
        cuDevMat temp(height - 1, width - 1, begin, pointer);
        cuComplex *temp_data = temp.get_data();
        cuComplex *res_data = res.get_data();
        for(int i=0; i < height;i++)
        {
            for(int j=0;j < width;j++)
            {
                for(int k=0;k < height - 1;k++)
                {
                    for(int t=0;t < width - 1;t++)
                    {
                        int row = k>=i?k+1:k;
                        int col = t>=j?t+1:t;
                        temp_data[k*(width - 1) + t] = data[row*width + col];
                    }
                } 
                cuComplex temp_det = cuCsubf(cuCmulf(temp_data[0], temp_data[3]), cuCmulf(temp_data[1], temp_data[2]));
                // cuComplex temp_det = temp.det();
                res_data[INDEX(j, i, width)] = cuCdivf(temp_det, det);
                if((i+j)%2 == 1)
                {
                    cuComplex alpha = make_cuComplex(-1, 0);
                    res_data[INDEX(j, i, width)] = cuCmulf(res_data[INDEX(j, i, width)], alpha);
                }
            }
        }
        temp.release();
        return res;
    }
}

__device__ cuDevMat cuDevMat::invParal()
{
    cuDevMat res;
    if(height != width)
    {
        printf("The matrix is not a square matrix\n");
        return res;
    }else{
        cuComplex mat_det;
        mat_det = this->det();
        if(cuCabsf(mat_det) < (float) 1e-5)
        {
            printf("the matrix is strange");          //矩阵奇异
            return res;
        }
        res.init(height, width, begin, pointer); 
        if(height == 1)
        {
            cuComplex temp = make_cuComplex(1, 0);
            res.set(0, 0, cuCdivf(temp, mat_det));
            return res;
        }
        int threadsize = ((height-1)*(width-1) + (width-1)*(width)*(2*width-1)/6 + 10)*sizeof(cuComplex);
        int InvSize = height*width*threadsize;  // 10 to insure enough space
        if((*pointer + InvSize) > THREADSPACE){
            printf("the matrix is too large and can be inversed in the threadspace\n");
            res.release();
            return res;
        }else{
            char *InvStart = (char *)begin + *pointer;
            dim3 blockdim(16, 16);
            dim3 griddim(1, 1);
            cuMatInv<<<griddim, blockdim>>>(this->get_data(), res.get_data(), height, width, mat_det, InvStart, threadsize);
            cudaDeviceSynchronize();
            return res;
        }
    }
}

__device__ cuDevMat cuDevMat::her(cuComplex alpha)
{
    cuDevMat res(height, height, begin, pointer);
    for(int i = 0;i < res.get_h();i++)
    {
        for(int j = 0;j < res.get_w();j++)
        {
            res.set(i, j, make_cuComplex(0, 0));
            for(int k = 0;k < width; k++)
            {
                res.set(i, j, cuCaddf(res.at(i, j), cuCmulf(data[i*width+k], cuConjf(data[j*width + k]))));    
            }
            res.set(i, j, cuCmulf(alpha, res.at(i, j)));
        }
    }
    return res;
}
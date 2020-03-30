#include"cuHostMat.h"
#include<stdio.h>

int main()
{
    cuHostMat temp(6, 6);
    int N = 6;
    cuComplex *a = new cuComplex[N*N], *b = new cuComplex[100];
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            a[INDEX(i, j, N)] = make_cuComplex(i, j);
        }
    }
    temp.init(a, N, N);
    cuHostMat temp1;
    temp.pad(temp1, 2, 2);
    // temp1.get_val(b);
    cout<<temp1<<endl;
    printf("%d", temp1.get_h());
    // cout<<temp.get_h()<<endl;
    // cout<<temp<<endl;
    return 0;
}
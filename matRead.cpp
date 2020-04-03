#include"matRead.h"

complex<float>*  matToArray(const char *filename, const char *variable){
    MATFile *pmatFile = NULL;
    mxArray *pMxArray = NULL;
    pmatFile = matOpen(filename , "r");
    if(pmatFile == NULL)
    {
        cout<<"mat file can not open"<<endl;
        exit(1);
    }
    pMxArray = matGetVariable(pmatFile, variable);
    const mwSize* array_dim = mxGetDimensions(pMxArray); // return Pointer of first element in the dimensions array
    mwSize num_dim = mxGetNumberOfDimensions(pMxArray); //  return number of dimensions in the specified mxArray(MATLAB Array)
    int h = (int)array_dim[0];
    int w = (int)array_dim[1];
    short  *ptr;
    ptr = (short *)mxGetInt16s(pMxArray);
    complex<float> *img_data = new complex<float>[h*w];
    float real,imag;
    // Matlab column priority storage
    for(int j=0;j<w;j++){
        for(int i=0;i<h;i++){
            real = (float)ptr[j*h+i];
            imag = (float)ptr[h*w + j*h + i];
            img_data[i*w + j] = complex<float>(real, imag);   // turn to row priority storage
        }
    }
    matClose(pmatFile);
    return img_data;
} 


/*----------------------------
getCudaInfo.cu:  C code to get CUDA capability and return info to MATLAB
Written by Ron Caplan
-------------------------------*/

#include "mex.h"

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions


/*Main mex function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int deviceCount;
    int cores_per_MP, totalcores,bestdev;
    double *mi;
    int dev=0;

    cudaDeviceProp deviceProp;
       
    cudaGetDeviceCount(&deviceCount);   
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA\n");
    }
    else{
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }   
    
    totalcores = 0;
    bestdev = 0;
    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        cores_per_MP = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        if(cores_per_MP*deviceProp.multiProcessorCount > totalcores){
             totalcores = cores_per_MP*deviceProp.multiProcessorCount;
             bestdev = dev;
        }
    }
    printf("Best device is device %d\n", bestdev);
    
    cudaSetDevice(bestdev);
    cudaGetDeviceProperties(&deviceProp, bestdev);
    
    printf("-----------------------------------------------------------------------\n");
    plhs[0] = mxCreateDoubleMatrix(1,2,mxREAL);

    mi = mxGetPr(plhs[0]);

    mi[0] = (double)deviceProp.major;
    mi[1] = (double)deviceProp.minor;    
    
    cores_per_MP = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
       

    printf("  CUDA device found.  Here are some specs:\n");
    printf("  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);
    printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
    printf("  Number of cores:                               %d\n", cores_per_MP * deviceProp.multiProcessorCount);
    printf("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    printf("-----------------------------------------------------------------\n");
    cudaDeviceReset();

 }

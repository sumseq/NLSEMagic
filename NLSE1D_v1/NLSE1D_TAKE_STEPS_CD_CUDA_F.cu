/*----------------------------
NLSE1D_TAKE_STEPS_CD_CUDA_F.cu
Program to integrate a chunk of time steps of the 1D Nonlinear Shrodinger Equation
i*Ut + a*Uxx - V(x)*U + s*|U|^2*U = 0
using RK4 + CD with CUDA compatable GPUs in
single precision.

Ronald M Caplan
Computational Science Research Center
San Diego State University

INPUT:
(U,V,s,a,h2,BC,chunk_size,k)
U  = Current solution matrix
V  = External Potential matrix
s  = Nonlinearity paramater
a  = Laplacian paramater
h2 = Spatial step size squared (h^2)
BC = Boundary condition selection switch:  1: Dirchilet 2: MSD 3: Lap=0 4: One-sided diff
chunk_size = Number of time steps to take
k  = Time step size

OUTPUT:
U:  New solution matrix
-------------------------------*/

#include "cuda.h"
#include "mex.h"
#include "math.h"

/*Define block size.*/

const int BLOCK_SIZE = 512;

/*Kernel to evaluate F(Psi) and take RK4 sub-step*/
__global__ void compute_F(float* ktotr, float* ktoti,
                          float* Utmpr, float* Utmpi,
                          float* Uoldr, float* Uoldi,
                          float* Uoutr, float* Uouti,
                          float* V, float s, float ah2,
                          int BC, int N, float k, int fstep)
{
    /*Declare shared memory space*/
    __shared__ float sUtmpr[BLOCK_SIZE+2];
    __shared__ float sUtmpi[BLOCK_SIZE+2];
    __shared__ float  NLSFr[BLOCK_SIZE+2];
    __shared__ float  NLSFi[BLOCK_SIZE+2];
    __shared__ float     sV[BLOCK_SIZE+2];

    /*Create two indexes:  one for shared, one for global*/
    int i  = blockIdx.x*blockDim.x + threadIdx.x;
    int si = threadIdx.x + 1;
    int msd_si;
    float OM;
    /*Copy vector from global memory into shared memory*/
    if(i<N)
    {
        sUtmpr[si] = Utmpr[i];
        sUtmpi[si] = Utmpi[i];
        sV[si]     =     V[i];
    }

    /*Synchronize the threads in the block so that all shared cells are filled.*/
    __syncthreads();

    /*If cell is not boundary...*/
    if (i > 0 && i < N-1)
    {
        /*Copy boundary layer of shared memory block*/
        if(si==1)
        {
            sUtmpr[si-1] = Utmpr[i-1];
            sUtmpi[si-1] = Utmpi[i-1];
        }
        if(si==blockDim.x)
        {
            sUtmpr[si+1] = Utmpr[i+1];
            sUtmpi[si+1] = Utmpi[i+1];
        }

        /*No synchthreads needed in this case*/
        NLSFr[si] = -ah2*(sUtmpi[si+1] - 2*sUtmpi[si] + sUtmpi[si-1])
                   -  (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpi[si];
        NLSFi[si] =  ah2*(sUtmpr[si+1] - 2*sUtmpr[si] + sUtmpr[si-1])
                   +  (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpr[si];
    }/*End of interier points*/

    if(BC==2)   __syncthreads(); /* needed for MSD*/

    /*Boundery Conditions*/
    if(i == 0 || i == N-1){
        switch (BC){
            case 1:
                NLSFr[si]   = 0.0f;
                NLSFi[si]   = 0.0f;
                break;
            case 2:
                if(i==0)     msd_si = si+1;
                if(i==N-1)   msd_si = si-1;

                OM = (NLSFi[msd_si]*sUtmpr[msd_si] - NLSFr[msd_si]*sUtmpi[msd_si])/
                    (sUtmpr[msd_si]*sUtmpr[msd_si] + sUtmpi[msd_si]*sUtmpi[msd_si]);

                NLSFr[si]  = -OM*sUtmpi[si];
                NLSFi[si]  =  OM*sUtmpr[si];
                break;
            case 3:
                NLSFr[si]   = - (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpi[si];
                NLSFi[si]   =   (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpr[si];
                break;
            case 4:
                if(i==0){
                NLSFr[si] = -ah2*(-Utmpi[3] + 4*Utmpi[2] - 5*sUtmpi[si+1] + 2*sUtmpi[si])
                            - (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpi[si];
                NLSFi[si] = ah2*(-Utmpr[3] + 4*Utmpr[2] - 5*sUtmpr[si+1] + 2*sUtmpr[si])
                            + (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpr[si];
                }
                else{
                NLSFr[si] = -ah2*(-Utmpi[N-4] + 4*Utmpi[N-3] - 5*sUtmpi[si-1] + 2*sUtmpi[si])
                            - (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si-1])*sUtmpi[si];
                NLSFi[si] =  ah2*(-Utmpr[N-4] + 4*Utmpr[N-3] - 5*sUtmpr[si-1] + 2*sUtmpr[si])
                            + (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si-1])*sUtmpr[si];
                }
                break;
            default:
                NLSFr[si]   = 0.0f;
                NLSFi[si]   = 0.0f;
                break;
       }/*BC Switch*/
    }/*BC*/

    if(i<N){
      switch(fstep)  {
          case 1:
            ktotr[i] = NLSFr[si];
            ktoti[i] = NLSFi[si];
            /*sUtmp is really Uold and Uold is really Utmp*/
            Uoldr[i] = sUtmpr[si] + k*NLSFr[si];
            Uoldi[i] = sUtmpi[si] + k*NLSFi[si];
            break;
          case 2:
            ktotr[i] = ktotr[i] + 2*NLSFr[si];
            ktoti[i] = ktoti[i] + 2*NLSFi[si];
            Uoutr[i] = Uoldr[i] + k*NLSFr[si];
            Uouti[i] = Uoldi[i] + k*NLSFi[si];
            break;
          case 3:
            Uoldr[i] = Uoldr[i] + k*(ktotr[i] + NLSFr[si]);
            Uoldi[i] = Uoldi[i] + k*(ktoti[i] + NLSFi[si]);
            break;
        }/*switch step*/
    }/*i<N*/

}/*Compute_F*/


/*Main mex function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int    i, j, N, chunk_size, BC;
    float  h2, a, ah2, s, k,k2,k6;
    double *Uoldr, *Uoldi, *V, *Unewr, *Unewi;

    float *fUoldr,*fUoldi,*fV;

    /*GPU variables:*/
    float *Uoldr_gpu, *Uoldi_gpu, *V_gpu;
    float *ktotr, *ktoti;
    float *Utmpr,*Utmpi;
    float *Uoutr,*Uouti;

    /* Find the dimensions of the vector */
    N = mxGetN(prhs[0]);
    if(N==1){
        N = mxGetM(prhs[0]);
    }

    /*Create result vector*/
    plhs[0] = mxCreateDoubleMatrix(N,1,mxCOMPLEX);

    /* Retrieve the input data */
    Uoldr = mxGetPr(prhs[0]);

    /*Check if initial condition fully real or not*/
    if(mxIsComplex(prhs[0])){
        Uoldi = mxGetPi(prhs[0]);
    }
    else{  /*Need to crate imaginary part vector*/
        Uoldi = (double*)malloc(N*sizeof(double));
        for(j=0;j<N;j++){
            Uoldi[j] = 0.0;
        }
    }

    /*Get the rest of the input variables*/
    V           = mxGetPr(prhs[1]);
    s           = (float)mxGetScalar(prhs[2]);
    a           = (float)mxGetScalar(prhs[3]);
    h2          = (float)mxGetScalar(prhs[4]);
    BC          =   (int)mxGetScalar(prhs[5]);
    chunk_size  =   (int)mxGetScalar(prhs[6]);
    k           = (float)mxGetScalar(prhs[7]);

    ah2 = a/h2; /*Saves some divisions*/
    k2  = k/2.0f;
    k6  = k/6.0f;


    /*Create cuda arrays on GPU*/
    cudaMalloc( (void **) &ktotr,sizeof(float)*N);
    cudaMalloc( (void **) &ktoti,sizeof(float)*N);
    cudaMalloc( (void **) &V_gpu,sizeof(float)*N);
    cudaMalloc( (void **) &Uoldr_gpu,sizeof(float)*N);
    cudaMalloc( (void **) &Uoldi_gpu,sizeof(float)*N);
    cudaMalloc( (void **) &Utmpr,sizeof(float)*N);
    cudaMalloc( (void **) &Utmpi,sizeof(float)*N);
    cudaMalloc( (void **) &Uoutr,sizeof(float)*N);
    cudaMalloc( (void **) &Uouti,sizeof(float)*N);

    /*Create float arrays on CPU*/
    fUoldr  = (float *) malloc(sizeof(float)*N);
    fUoldi  = (float *) malloc(sizeof(float)*N);
    fV      = (float *) malloc(sizeof(float)*N);

    /*Convert initial condition to float type:*/
    for(i=0;i<N;i++)
    {
        fUoldr[i] = (float)Uoldr[i];
        fUoldi[i] = (float)Uoldi[i];
            fV[i] = (float)V[i];
    }

    /*Copy initial condition and potential vectors to GPU*/
    cudaMemcpy( Uoldr_gpu, fUoldr, sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaMemcpy( Uoldi_gpu, fUoldi, sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaMemcpy( V_gpu,         fV, sizeof(float)*N,cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)ceil((N+0.0)/dimBlock.x));

    /*Compute chunk of time steps using RK4*/
    for (j = 0; j<chunk_size; j++)
    {
      compute_F   <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoldr_gpu,Uoldi_gpu,Utmpr,    Utmpi,    V_gpu,V_gpu,V_gpu,s,ah2,BC,N,k2,1);
      compute_F   <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,Uoutr,Uouti,V_gpu,s,ah2,BC,N,k2,2);
      compute_F   <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoutr,    Uouti,    Uoldr_gpu,Uoldi_gpu,Utmpr,Utmpi,V_gpu,s,ah2,BC,N,k, 2);
      compute_F   <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,V_gpu,V_gpu,V_gpu,s,ah2,BC,N,k6,3);
    }
    
    /*Get pointers to result vectors*/
    Unewr = mxGetPr(plhs[0]);
    Unewi = mxGetPi(plhs[0]);
    
    /*Make sure everything is done (important for large chunk-size computations)*/
    cudaDeviceSynchronize();
    
    /*Copy result from GPU back to CPU*/
    cudaMemcpy( fUoldr, Uoldr_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy( fUoldi, Uoldi_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost);

    /*Convert result to double type*/
    for(i=0;i<N;i++){
        Unewr[i] = (double)fUoldr[i];
        Unewi[i] = (double)fUoldi[i];
    }

    /*Free up GPU memory*/
    cudaFree(ktotr);
    cudaFree(ktoti);
    cudaFree(V_gpu);
    cudaFree(Uoldr_gpu);
    cudaFree(Uoldi_gpu);
    cudaFree(Utmpr);
    cudaFree(Utmpi);
    cudaFree(Uoutr);
    cudaFree(Uouti);

    /*Free up CPU memory*/
    free(fUoldr);
    free(fUoldi);
    free(fV);

    if(!mxIsComplex(prhs[0])){
        free(Uoldi);
    }

    cudaDeviceReset();
}

/*For reference, command to compile code in MATLAB on windows:
nvmex -f nvmexopts.bat NLSE1D_TAKE_STEPS_CD_CUDA_F.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
*/

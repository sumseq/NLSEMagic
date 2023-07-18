/*----------------------------
NLSE1D_TAKE_STEPS_2SHOC_CUDA_D.cu
Program to integrate a chunk of time steps of the 1D Nonlinear Shrodinger Equation
i*Ut + a*Uxx - V(x)*U + s*|U|^2*U = 0
using RK4 + 2SHOC with CUDA compatable GPUs in
double precision.

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

/*Kernel to evaluate D(Psi) using shared memory*/
__global__ void compute_D  (double* Dr,    double* Di,
                            double* Utmpr, double* Utmpi,
                            double* V,
                            double lh2,  double l_a, double s,int BC, int N)
{
    /*Declare shared memory space*/
    __shared__ double sUtmpr[BLOCK_SIZE+2];
    __shared__ double sUtmpi[BLOCK_SIZE+2];
    /*Create two indexes:  one for shared, one for global*/
    int i  = blockIdx.x*blockDim.x+threadIdx.x;
    int si = threadIdx.x+1;
    double A,Nb,Nb1;
    int msd_si,msd_i;

    /*Copy vector from global memory into shared memory*/
    if(i<N)
    {
        sUtmpr[si] = Utmpr[i];
        sUtmpi[si] = Utmpi[i];
    }
    /*Synchronize the threads in the block so that all shared cells are filled.*/
    __syncthreads();

    /*If cell is not boundary...*/
    if (i >0 && i < N-1)
    {
        if(si==1)
        {
            sUtmpr[0] = Utmpr[i-1];
            sUtmpi[0] = Utmpi[i-1];
        }
        if(si==blockDim.x)
        {
            sUtmpr[si+1] = Utmpr[i+1];
            sUtmpi[si+1] = Utmpi[i+1];
        }
        /*No synchthreads needed in this case*/
        Dr[i] = (sUtmpr[si+1] - 2*sUtmpr[si] + sUtmpr[si-1])*lh2;
        Di[i] = (sUtmpi[si+1] - 2*sUtmpi[si] + sUtmpi[si-1])*lh2;
    }

    if(BC==2)   __syncthreads(); /* needed for MSD*/

        /*Boundery Conditions*/
    if(i == 0 || i == N-1){

        /*Boundary conditions:*/
        switch (BC){
            case 1: /*Dirichlet*/
                Dr[i] = -l_a*(s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - V[i])*sUtmpr[si];
                Di[i] = -l_a*(s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - V[i])*sUtmpi[si];
                break;
            case 2: /*MSD*/
                if(i==0)                msd_si = si+1;
                if(i==N-1)              msd_si = si-1;
                msd_i = i+(msd_si-si);

                Nb  = s*(sUtmpr[si]*sUtmpr[si]         + sUtmpi[si]*sUtmpi[si])         - V[i];
                Nb1 = s*(sUtmpr[msd_si]*sUtmpr[msd_si] + sUtmpi[msd_si]*sUtmpi[msd_si]) - V[msd_i];

                A   = (Dr[msd_i]*sUtmpr[msd_si]      + Di[msd_i]*sUtmpi[msd_si])/
                      (sUtmpr[msd_si]*sUtmpr[msd_si] + sUtmpi[msd_si]*sUtmpi[msd_si]);
                    
                Dr[i]  = (A + l_a*(Nb1-Nb))*sUtmpr[si];
                Di[i]  = (A + l_a*(Nb1-Nb))*sUtmpi[si];
                break;
            case 3: /*L0*/
                Dr[i]   = 0.0;
                Di[i]   = 0.0;
                break;
            case 4: /*1-sided*/
                if(i==0){
                   Dr[i] = (-Utmpr[3] + 4*Utmpr[2] - 5*sUtmpr[si+1] + 2*sUtmpr[si])*lh2;
                   Di[i] = (-Utmpi[3] + 4*Utmpi[2] - 5*sUtmpi[si+1] + 2*sUtmpi[si])*lh2;
                }
                else{
                   Dr[i] = (-Utmpr[N-4] + 4*Utmpr[N-3] - 5*sUtmpr[si-1] + 2*sUtmpr[si])*lh2;
                   Di[i] = (-Utmpi[N-4] + 4*Utmpi[N-3] - 5*sUtmpi[si-1] + 2*sUtmpi[si])*lh2;
                }
                break;
            default:
                Dr[i]   = 0.0;
                Di[i]   = 0.0;
                break;
       }/*BC Switch*/
    }/*BC*/
}/*computedx2*/


/*Kernel to evaluate F(Psi) using shared memory*/
__global__ void compute_F(double* ktotr, double* ktoti,
                          double* Utmpr, double* Utmpi,
                          double* Uoldr, double* Uoldi,
                          double* Uoutr, double* Uouti,
                          double* Dr,    double* Di,
                          double* V, double s, double a,double a76,double a112,
                          int BC, int N, double k, int fstep)
{
    /*Declare shared memory space*/
    __shared__ double sUtmpr[BLOCK_SIZE+2];
    __shared__ double sUtmpi[BLOCK_SIZE+2];
    __shared__ double  NLSFr[BLOCK_SIZE+2];
    __shared__ double  NLSFi[BLOCK_SIZE+2];
    __shared__ double    sDr[BLOCK_SIZE+2];
    __shared__ double    sDi[BLOCK_SIZE+2];
    __shared__ double     sV[BLOCK_SIZE+2];

    /*Create two indexes:  one for shared, one for global*/
    int i  = blockIdx.x*blockDim.x+threadIdx.x;
    int si = threadIdx.x+1;
    double OM;
    int msd_si;

    /*Copy vectors from global memory into shared memory*/
    if(i<N)
    {
        sUtmpr[si] = Utmpr[i];
        sUtmpi[si] = Utmpi[i];
        sDr[si]    = Dr[i];
        sDi[si]    = Di[i];
        sV[si]     = V[i];
    }

    /*Synchronize the threads in the block so that all shared cells are filled.*/
    __syncthreads();

    /*If cell is not boundary...*/
    if (i >0 && i< N-1)
    {
        if(si==1)
        {
           sDr[0] =  Dr[i-1];
           sDi[0] =  Di[i-1];
        }
        if(si==blockDim.x)
        {
           sDr[si+1] = Dr[i+1];
           sDi[si+1] = Di[i+1];
        }
        /*No synchthreads needed in this case*/
        NLSFr[si] = a112*(sDi[si+1] + sDi[si-1]) - a76*sDi[si]
                 - (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpi[si];
        NLSFi[si] =  a76*sDr[si] - a112*(sDr[si+1] + sDr[si-1])
                 + (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpr[si];
    }/*End of interier points*/

    if(BC==2)   __syncthreads(); /* needed for MSD*/

    /*Boundery Conditions*/
    if(i == 0 || i == N-1){

        /*Boundary conditions:*/
        switch (BC){
            case 1:
                NLSFr[si]   = 0.0;
                NLSFi[si]   = 0.0;
                break;
            case 2:
                if(i==0)     msd_si = si+1;
                if(i==N-1)   msd_si = si-1;

                OM = (NLSFi[msd_si]*sUtmpr[msd_si]  - NLSFr[msd_si]*sUtmpi[msd_si])/
                     (sUtmpr[msd_si]*sUtmpr[msd_si] + sUtmpi[msd_si]*sUtmpi[msd_si]);
                                        
                NLSFr[si]  = -OM*sUtmpi[si];
                NLSFi[si]  =  OM*sUtmpr[si];
                break;
            case 3:
                NLSFr[si]   = - (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpi[si];
                NLSFi[si]   =   (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpr[si];
                break;
            case 4:
                NLSFr[si] = -a*sDi[si] - (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpi[si];
                NLSFi[si] =  a*sDr[si] + (s*(sUtmpr[si]*sUtmpr[si] + sUtmpi[si]*sUtmpi[si]) - sV[si])*sUtmpr[si];
                break;
            default:
                NLSFr[si]   = 0.0;
                NLSFi[si]   = 0.0;
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
    int     j, N, chunk_size, BC;
    double  h2, lh2, l_a, a, a76, a112, s, k, k2, k6;
    double *Uoldr, *Uoldi, *V, *Unewr, *Unewi;

    /*GPU variables:*/
    double *Uoldr_gpu, *Uoldi_gpu, *V_gpu;
    double *Uoutr,*Uouti, *ktotr, *ktoti;
    double *Utmpr,*Utmpi,*Dr, *Di;

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
    s           = (double)mxGetScalar(prhs[2]);
    a           = (double)mxGetScalar(prhs[3]);
    h2          = (double)mxGetScalar(prhs[4]);
    BC          =    (int)mxGetScalar(prhs[5]);
    chunk_size  =    (int)mxGetScalar(prhs[6]);
    k           = (double)mxGetScalar(prhs[7]);

    /*Create cuda arrays on GPU*/
    cudaMalloc( (void **) &Uoutr,    sizeof(double)*N);
    cudaMalloc( (void **) &Uouti,    sizeof(double)*N);
    cudaMalloc( (void **) &ktotr,    sizeof(double)*N);
    cudaMalloc( (void **) &ktoti,    sizeof(double)*N);
    cudaMalloc( (void **) &V_gpu,    sizeof(double)*N);
    cudaMalloc( (void **) &Uoldr_gpu,sizeof(double)*N);
    cudaMalloc( (void **) &Uoldi_gpu,sizeof(double)*N);
    cudaMalloc( (void **) &Utmpr,    sizeof(double)*N);
    cudaMalloc( (void **) &Utmpi,    sizeof(double)*N);
    cudaMalloc( (void **) &Dr,       sizeof(double)*N);
    cudaMalloc( (void **) &Di,       sizeof(double)*N);

    /*Copy initial condition and potential vectors to GPU*/
    cudaMemcpy( Uoldr_gpu, Uoldr, sizeof(double)*N,cudaMemcpyHostToDevice);
    cudaMemcpy( Uoldi_gpu, Uoldi, sizeof(double)*N,cudaMemcpyHostToDevice);
    cudaMemcpy( V_gpu,         V, sizeof(double)*N,cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)ceil((N+0.0)/dimBlock.x));

    l_a  = 1.0/a;
    lh2  = 1.0/h2;
    k2   = k/2.0;
    k6   = k/6.0;
    a76  = a*(7.0/6.0);
    a112 = a*(1.0/12.0);

    /*Compute chunk of time steps using RK4*/
    for (j = 0; j<chunk_size; j++)
    {
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Uoldr_gpu,Uoldi_gpu,V_gpu,lh2,l_a,s,BC,N);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoldr_gpu,Uoldi_gpu,Utmpr,    Utmpi,    V_gpu,V_gpu,Dr,Di,V_gpu,s,a,a76,a112,BC,N,k2,1);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Utmpr,Utmpi,V_gpu,lh2,l_a,s,BC,N);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,Uoutr,Uouti,Dr,Di,V_gpu,s,a,a76,a112,BC,N,k2,2);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Uoutr,Uouti,V_gpu,lh2,l_a,s,BC,N);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoutr,    Uouti,    Uoldr_gpu,Uoldi_gpu,Utmpr,Utmpi,Dr,Di,V_gpu,s,a,a76,a112,BC,N,k, 2);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Utmpr,Utmpi,V_gpu,lh2,l_a,s,BC,N);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,V_gpu,V_gpu,Dr,Di,V_gpu,s,a,a76,a112,BC,N,k6,3);
     }

    /*Get pointers to result vectors*/
    Unewr = mxGetPr(plhs[0]);
    Unewi = mxGetPi(plhs[0]);
    
    /*Make sure everything is done (important for large chunk-size computations)*/
    cudaDeviceSynchronize();

    /*Copy result from GPU back to CPU*/
    cudaMemcpy(Unewr, Uoldr_gpu, sizeof(double)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(Unewi, Uoldi_gpu, sizeof(double)*N, cudaMemcpyDeviceToHost);

    /*Free up GPU memory*/
    cudaFree(Uoutr);
    cudaFree(Uouti);
    cudaFree(ktotr);
    cudaFree(ktoti);
    cudaFree(V_gpu);
    cudaFree(Uoldr_gpu);
    cudaFree(Uoldi_gpu);
    cudaFree(Utmpr);
    cudaFree(Utmpi);
    cudaFree(Dr);
    cudaFree(Di);

    if(!mxIsComplex(prhs[0])){
     free(Uoldi);
    }

    cudaDeviceReset();
}

/*For reference, command to compile code in MATLAB on windows:
nvmex -f nvmexopts.bat NLSE1D_TAKE_STEPS_2SHOC_CUDA_D.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
*/


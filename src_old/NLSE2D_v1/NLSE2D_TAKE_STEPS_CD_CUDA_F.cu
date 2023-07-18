/*----------------------------
NLSE2D_TAKE_STEPS_CD_CUDA_F.cu
Program to integrate a chunk of time steps of the 2D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy) - V(x,y)*U + s*|U|^2*U = 0
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
BC = Boundary condition selection switch:  1: Dirchilet 2:MSD 3:Lap=0
chunk_size = Number of time steps to take
k  = Time step size

OUTPUT:
U:  New solution matrix
-------------------------------*/

#include "cuda.h"
#include "mex.h"
#include "math.h"

/*Define block size*/
const int BLOCK_SIZEX = 16;
const int BLOCK_SIZEY = 16;

/*Kernel to evaluate F(Psi) using shared memory*/
__global__ void compute_F(float* ktotr, float* ktoti,
                          float* Utmpr, float* Utmpi,
                          float* Uoldr, float* Uoldi,
                          float* Uoutr, float* Uouti,
                          float* V,     float s,     float ah2,
                          int BC, int N, int M, int pitch, float k, int fstep)
{
    /*Declare shared memory space*/
    __shared__ float sUtmpr[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float sUtmpi[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float  NLSFr[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float  NLSFi[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float     sV[BLOCK_SIZEY+2][BLOCK_SIZEX+2];

    /*Create four indexes:  two for shared, two for global*/
    int j   = blockIdx.x*blockDim.x+threadIdx.x;
    int sj  = threadIdx.x + 1;
    int i   = blockIdx.y*blockDim.y+threadIdx.y;
    int si  = threadIdx.y + 1;

    int msd_si, msd_sj;
    float OM;

    int ij = pitch*i+j;

    if(i<N && j<M)
    {
        /*Copy blocksized matrix from global memory into shared memory*/
        sUtmpr[si][sj] = Utmpr[ij];
        sUtmpi[si][sj] = Utmpi[ij];
            sV[si][sj] =     V[ij];
    }

    /*Synchronize the threads in the block so that all shared cells are filled.*/
    __syncthreads();

    /*If cell is NOT on boundary...*/
    if ((j>0 && j<M-1)&&(i>0 && i<N-1))
    {
        /*Copy boundary layer of shared memory block*/
        if(si==1)
        {
            sUtmpr[si-1][sj] = Utmpr[ij-pitch];
            sUtmpi[si-1][sj] = Utmpi[ij-pitch];
        }
        if(si==blockDim.y)
        {
            sUtmpr[si+1][sj] = Utmpr[ij+pitch];
            sUtmpi[si+1][sj] = Utmpi[ij+pitch];
        }
        if(sj==1)
        {
            sUtmpr[si][sj-1] = Utmpr[ij-1];
            sUtmpi[si][sj-1] = Utmpi[ij-1];
        }
        if(sj==blockDim.x)
        {
            sUtmpr[si][sj+1] = Utmpr[ij+1];
            sUtmpi[si][sj+1] = Utmpi[ij+1];
        }
         /*No synchthreads needed in this case*/
        NLSFr[si][sj] = -ah2*(sUtmpi[si+1][sj] - 4*sUtmpi[si][sj] + sUtmpi[si-1][sj] +
                              sUtmpi[si][sj+1]                    + sUtmpi[si][sj-1])
                            -(s*(sUtmpr[si][sj]*sUtmpr[si][sj] +
                                 sUtmpi[si][sj]*sUtmpi[si][sj]) - sV[si][sj])*sUtmpi[si][sj];

        NLSFi[si][sj] =  ah2*(sUtmpr[si+1][sj] - 4*sUtmpr[si][sj] + sUtmpr[si-1][sj] +
                              sUtmpr[si][sj+1]                    + sUtmpr[si][sj-1])
                            +(s*(sUtmpr[si][sj]*sUtmpr[si][sj] +
                                 sUtmpi[si][sj]*sUtmpi[si][sj]) - sV[si][sj])*sUtmpr[si][sj];
    }/*End of interier points*/

    if(BC==2)   __syncthreads(); /* needed for MSD*/

    if(i<N && j<M)
    {
        /*Cell is ON Boundery*/
        if(!((j>0 && j<M-1)&&(i>0 && i<N-1)))
        {
            switch(BC){
                case 1: /*Dirichlet*/
                    NLSFr[si][sj]   = 0.0f;
                    NLSFi[si][sj]   = 0.0f;
                    break;
                case 2: /* Mod-Squared Dirichlet |U|^2=B */
                    if(i==0)                msd_si = si+1;
                    if(i==N-1)              msd_si = si-1;
                    if((i!=0) && (i!=N-1))  msd_si = si;
                    if(j==0)                msd_sj = sj+1;
                    if(j==M-1)              msd_sj = sj-1;
                    if((j!=0) && (j!=M-1))  msd_sj = sj;

                    OM = (NLSFi[msd_si][msd_sj]*sUtmpr[msd_si][msd_sj] - NLSFr[msd_si][msd_sj]*sUtmpi[msd_si][msd_sj])/
                         (sUtmpr[msd_si][msd_sj]*sUtmpr[msd_si][msd_sj] + sUtmpi[msd_si][msd_sj]*sUtmpi[msd_si][msd_sj]);
                                        
                    NLSFr[si][sj]  = -OM*sUtmpi[si][sj];
                    NLSFi[si][sj]  =  OM*sUtmpr[si][sj];
                    break;
                case 3: /*Uxx+Uyy = 0:*/
                    NLSFr[si][sj] = - (s*(sUtmpr[si][sj]*sUtmpr[si][sj] + sUtmpi[si][sj]*sUtmpi[si][sj]) - sV[si][sj])*sUtmpi[si][sj];
                    NLSFi[si][sj] =   (s*(sUtmpr[si][sj]*sUtmpr[si][sj] + sUtmpi[si][sj]*sUtmpi[si][sj]) - sV[si][sj])*sUtmpr[si][sj];
                    break;
                default:
                    NLSFr[si][sj]   = 0.0f;
                    NLSFi[si][sj]   = 0.0f;
                    break;
            }/*BC switch*/
        }/*on BC*/

        switch(fstep)  {
          case 1:
            ktotr[ij] = NLSFr[si][sj];
            ktoti[ij] = NLSFi[si][sj];
            /*sUtmp is really Uold and Uold is really Utmp*/
            Uoldr[ij] = sUtmpr[si][sj] + k*NLSFr[si][sj];
            Uoldi[ij] = sUtmpi[si][sj] + k*NLSFi[si][sj];
            break;
          case 2:
            ktotr[ij] = ktotr[ij] + 2*NLSFr[si][sj];
            ktoti[ij] = ktoti[ij] + 2*NLSFi[si][sj];
            Uoutr[ij] = Uoldr[ij] + k*NLSFr[si][sj];
            Uouti[ij] = Uoldi[ij] + k*NLSFi[si][sj];
            break;
          case 3:
            Uoldr[ij] = Uoldr[ij] + k*(ktotr[ij] + NLSFr[si][sj]);
            Uoldi[ij] = Uoldi[ij] + k*(ktoti[ij] + NLSFi[si][sj]);
            break;
        }/*switch step*/

    }/*<end*/
}/*Compute_F*/

/*Main mex function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int    i, j, N, M, chunk_size, BC;
    float  h2, a, ah2, s, k, k2, k6;
    double *vUoldr, *vUoldi,  *vV;
    double *vUnewr, *vUnewi;
    float  *fvUoldr, *fvUoldi,  *fvV;
    size_t pitch;
    int   ipitch;

    /*GPU variables:*/
    float *Uoldr_gpu, *Uoldi_gpu, *V_gpu;
    float *ktotr, *ktoti;
    float *Utmpr,*Utmpi;
    float *Uoutr,*Uouti;

    /*Find the dimensions of the input matrix*/
    N = mxGetN(prhs[0]);
    M = mxGetM(prhs[0]);

    /*Create output vector*/
    plhs[0] = mxCreateDoubleMatrix(M,N,mxCOMPLEX);

    /*Retrieve the input data*/
    vUoldr = mxGetPr(prhs[0]);
    if(mxIsComplex(prhs[0])){
        vUoldi = mxGetPi(prhs[0]);
    }
    else{
        vUoldi = (double *)malloc(sizeof(double)*N*M);
        for(i=0;i<N*M;i++){
            vUoldi[i] = 0.0;
        }
    }
    vV     = mxGetPr(prhs[1]);

    /*Get the rest of the input variables*/
    s           = (float)mxGetScalar(prhs[2]);
    a           = (float)mxGetScalar(prhs[3]);
    h2          = (float)mxGetScalar(prhs[4]);
    BC          =   (int)mxGetScalar(prhs[5]);
    chunk_size  =   (int)mxGetScalar(prhs[6]);
    k           = (float)mxGetScalar(prhs[7]);

    /*Pre-compute parameter divisions*/
    ah2 = a/h2;
    k2  = k/2.0f;
    k6  = k/6.0f;

    /*Allocate float input vectors*/
    fvUoldr = (float *)malloc(sizeof(float)*N*M);
    fvUoldi = (float *)malloc(sizeof(float)*N*M);
    fvV     = (float *)malloc(sizeof(float)*N*M);

    /*Allocate 2D CUDA memory*/
    cudaMallocPitch((void**) &Uoldr_gpu, &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &Uoldi_gpu, &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &V_gpu,     &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &Utmpr,     &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &Utmpi,     &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &Uouti,     &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &Uoutr,     &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &ktotr,     &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &ktoti,     &pitch, M*sizeof(float), N);

    /*Convert double input vectors to float*/
    for(i=0;i<N*M;i++)
    {
        fvUoldr[i] = (float)vUoldr[i];
        fvUoldi[i] = (float)vUoldi[i];
            fvV[i] = (float)vV[i];
    }

    /*Copy input vectors to GPU*/
    cudaMemcpy2D(Uoldr_gpu,pitch, fvUoldr,M*sizeof(float),M*sizeof(float),N,cudaMemcpyHostToDevice);
    cudaMemcpy2D(Uoldi_gpu,pitch, fvUoldi,M*sizeof(float),M*sizeof(float),N,cudaMemcpyHostToDevice);
    cudaMemcpy2D(V_gpu,    pitch, fvV,    M*sizeof(float),M*sizeof(float),N,cudaMemcpyHostToDevice);

    /*Compute index for pitch vector memory*/
    ipitch = (int)(pitch/sizeof(float));

    /*Set up CUDA grid and block size*/
    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY);
    dim3 dimGrid((int)ceil((M+0.0)/dimBlock.x), (int)ceil((N+0.0)/dimBlock.y));

    /*Compute chunk of time steps*/
    for (j = 0; j<chunk_size; j++)
    {
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoldr_gpu,Uoldi_gpu,Utmpr,    Utmpi,    V_gpu,V_gpu,V_gpu,s,ah2,BC,N,M,ipitch,k2,1);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,Uoutr,Uouti,V_gpu,s,ah2,BC,N,M,ipitch,k2,2);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoutr,    Uouti,    Uoldr_gpu,Uoldi_gpu,Utmpr,Utmpi,V_gpu,s,ah2,BC,N,M,ipitch,k, 2);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,V_gpu,V_gpu,V_gpu,s,ah2,BC,N,M,ipitch,k6,3);
    }

    /*Set up output vectors*/
    vUnewr = mxGetPr(plhs[0]);
    vUnewi = mxGetPi(plhs[0]);
    
    /*Make sure everything is done (important for large chunk-size computations)*/
    cudaDeviceSynchronize();

    /*Transfer solution back to CPU*/
    cudaMemcpy2D(fvUoldr,M*sizeof(float),Uoldr_gpu,pitch,M*sizeof(float),N,cudaMemcpyDeviceToHost);
    cudaMemcpy2D(fvUoldi,M*sizeof(float),Uoldi_gpu,pitch,M*sizeof(float),N,cudaMemcpyDeviceToHost);

    /*Convert float vector to double output vector*/
    for(i=0;i<N*M;i++){
        vUnewr[i] = (double)fvUoldr[i];
        vUnewi[i] = (double)fvUoldi[i];
    }

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

    /*Free up CPU memory*/
    free(fvUoldr);
    free(fvUoldi);
    free(fvV);

    if(!mxIsComplex(prhs[0])){
        free(vUoldi);
    }

    cudaDeviceReset();
}

/*For reference, command to compile code in MATLAB on windows:
nvmex -f nvmexopts.bat NLSE2D_TAKE_STEPS_CD_CUDA_F.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
*/

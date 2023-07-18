/*----------------------------
NLSE3D_TAKE_STEPS_CD_CUDA_D.cu:
Program to integrate a chunk of time steps of the 3D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy + Uzz) - V(r)*U + s*|U|^2*U = 0
using RK4 + CD with CUDA compatable GPUs in
double precision.

Ronald N Caplan
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
const int BLOCK_SIZEX = 10;
const int BLOCK_SIZEY = 10;
const int BLOCK_SIZEZ = 5;

/*Kernel to evaluate F(Psi) using shared memory*/
__global__ void compute_F(double* ktotr, double* ktoti,
                          double* Utmpr, double* Utmpi,
                          double* Uoldr, double* Uoldi,
                          double* Uoutr, double* Uouti,
                          double* V, double s, double ah2,
                          int BC, int L, int N, int M, int gridDim_y, double K, int fstep)
{
    /*Declare shared memory space*/
    __shared__ double sUtmpr[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ double sUtmpi[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ double  NLSFr[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ double  NLSFi[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ double     sV[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];

    /*Create six indexes:  three for shared, three for global*/
    int i, j, k, blockIdx_z, blockIdx_y;
    /*Compute idx for z in cube (int division acts as floor operator here)*/
    blockIdx_z  = blockIdx.y/gridDim_y;
    /*Compute "true" idx for y in cube*/
    blockIdx_y  = blockIdx.y - blockIdx_z*gridDim_y;
    /*Now can compute j and k as if there was a 3D CUDA grid:*/
    k = blockIdx.x*blockDim.x + threadIdx.x;
    j = blockIdx_y*blockDim.y + threadIdx.y;
    i = blockIdx_z*blockDim.z + threadIdx.z;

    int sk  = threadIdx.x + 1;
    int sj  = threadIdx.y + 1;
    int si  = threadIdx.z + 1;

    int msd_si, msd_sj, msd_sk;
    double OM;

    int ijk = N*M*i + M*j + k;

    if(i<L && (j<N && k<M)){
        /*Copy blocksized matrix from global memory into shared memory*/
        sUtmpr[si][sj][sk] = Utmpr[ijk];
        sUtmpi[si][sj][sk] = Utmpi[ijk];
            sV[si][sj][sk] =     V[ijk];
    }

    /*Synchronize the threads in the block so that all shared cells are filled.*/
    __syncthreads();

    /*If cell is NOT on boundary...*/
    if ( (j>0 && j<N-1) && ((i>0 && i<L-1) && (k>0 && k<M-1)) )
    {
        /*Copy boundary layer of shared memory block*/
        if(si==1)
        {
            sUtmpr[0][sj][sk] = Utmpr[ijk - N*M];
            sUtmpi[0][sj][sk] = Utmpi[ijk - N*M];
        }
        if(si==blockDim.z)
        {
            sUtmpr[si+1][sj][sk] = Utmpr[ijk + N*M];
            sUtmpi[si+1][sj][sk] = Utmpi[ijk + N*M];
        }
        if(sj==1)
        {
            sUtmpr[si][0][sk] = Utmpr[ijk - M];
            sUtmpi[si][0][sk] = Utmpi[ijk - M];
        }
        if(sj==blockDim.y)
        {
            sUtmpr[si][sj+1][sk] = Utmpr[ijk + M];
            sUtmpi[si][sj+1][sk] = Utmpi[ijk + M];
        }
        if(sk==1)
        {
            sUtmpr[si][sj][0] = Utmpr[ijk - 1];
            sUtmpi[si][sj][0] = Utmpi[ijk - 1];
        }
        if(sk==blockDim.x)
        {
            sUtmpr[si][sj][sk+1] = Utmpr[ijk + 1];
            sUtmpi[si][sj][sk+1] = Utmpi[ijk + 1];
        }

        /*No synchthreads needed in this case*/

        NLSFr[si][sj][sk] = -ah2*(sUtmpi[si+1][sj][sk] - 6*sUtmpi[si][sj][sk] + sUtmpi[si-1][sj][sk] +
                           sUtmpi[si][sj+1][sk]                        + sUtmpi[si][sj-1][sk] +
                           sUtmpi[si][sj][sk+1]                        + sUtmpi[si][sj][sk-1])
                         + (sV[si][sj][sk] - s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] +
                                        sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]))*sUtmpi[si][sj][sk];

        NLSFi[si][sj][sk] =  ah2*(sUtmpr[si+1][sj][sk] - 6*sUtmpr[si][sj][sk] + sUtmpr[si-1][sj][sk] +
                           sUtmpr[si][sj+1][sk]                        + sUtmpr[si][sj-1][sk] +
                           sUtmpr[si][sj][sk+1]                        + sUtmpr[si][sj][sk-1])
                         - (sV[si][sj][sk] - s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] +
                                       sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]))*sUtmpr[si][sj][sk];

    }/*End of interier points*/

    /*This synch is needed for the MSD boundary condition*/
    if(BC==2)  __syncthreads();

    if(i<L && (j<N && k<M)){

        /*Cell is ON Boundery*/
        if(!( (j>0 && j<N-1) && ((i>0 && i<L-1) && (k>0 && k<M-1)) ) ){
            switch(BC){
                case 1: /*Dirichlet*/
                    NLSFr[si][sj][sk]   = 0.0;
                    NLSFi[si][sj][sk]   = 0.0;
                    break;
                case 2: /* Mod-Squared Dirichlet |U|^2=B */
                    if(i==0)                msd_si = si+1;
                    if(i==L-1)              msd_si = si-1;
                    if((i!=0) && (i!=L-1))  msd_si = si;
                    if(j==0)                msd_sj = sj+1;
                    if(j==N-1)              msd_sj = sj-1;
                    if((j!=0) && (j!=N-1))  msd_sj = sj;
                    if(k==0)                msd_sk = sk+1;
                    if(k==M-1)              msd_sk = sk-1;
                    if((k!=0) && (k!=M-1))  msd_sk = sk;

                    if(sUtmpr[msd_si][msd_sj][msd_sk]==0 && sUtmpi[msd_si][msd_sj][msd_sk]==0)
                    {
                        OM=0;
                    }
                    else{ 
                        OM = (NLSFi[msd_si][msd_sj][msd_sk]*sUtmpr[msd_si][msd_sj][msd_sk] -  NLSFr[msd_si][msd_sj][msd_sk]*sUtmpi[msd_si][msd_sj][msd_sk])/
                         (sUtmpr[msd_si][msd_sj][msd_sk]*sUtmpr[msd_si][msd_sj][msd_sk] + sUtmpi[msd_si][msd_sj][msd_sk]*sUtmpi[msd_si][msd_sj][msd_sk]);
                    }                    
                    NLSFr[si][sj][sk]  = -OM*sUtmpi[si][sj][sk];
                    NLSFi[si][sj][sk]  =  OM*sUtmpr[si][sj][sk];

                    break;
                case 3: /*Uxx+Uyy+Uzz=0:*/
                    NLSFr[si][sj][sk] = - (s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] + sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]) - sV[si][sj][sk])*sUtmpi[si][sj][sk];
                    NLSFi[si][sj][sk] =   (s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] + sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]) - sV[si][sj][sk])*sUtmpr[si][sj][sk];
                    break;
                default:
                    NLSFr[si][sj][sk]   = 0.0;
                    NLSFi[si][sj][sk]   = 0.0;
                    break;
            }/*BC switch*/
        }/*on BC*/

        switch(fstep)  {
          case 1:
            ktotr[ijk] = NLSFr[si][sj][sk];
            ktoti[ijk] = NLSFi[si][sj][sk];
            /*sUtmp is really Uold and Uold is really Utmp*/
            Uoldr[ijk] = sUtmpr[si][sj][sk] + K*NLSFr[si][sj][sk];
            Uoldi[ijk] = sUtmpi[si][sj][sk] + K*NLSFi[si][sj][sk];
            break;
          case 2:
            ktotr[ijk] = ktotr[ijk] + 2*NLSFr[si][sj][sk];
            ktoti[ijk] = ktoti[ijk] + 2*NLSFi[si][sj][sk];
            Uoutr[ijk] = Uoldr[ijk] + K*NLSFr[si][sj][sk];
            Uouti[ijk] = Uoldi[ijk] + K*NLSFi[si][sj][sk];
            break;
          case 3:
            Uoldr[ijk] = Uoldr[ijk] + K*(ktotr[ijk] + NLSFr[si][sj][sk]);
            Uoldi[ijk] = Uoldi[ijk] + K*(ktoti[ijk] + NLSFi[si][sj][sk]);
            break;

        }/*switch step*/
    }/*<end*/
}/*Compute_F*/

/*Main mex function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int    L,N,M,dims,gridDim_y;
    const int *dim_array;
    int    i, c, chunk_size, BC;
    double  h2, a, ah2, s, K, K2, K6;
    double *vUoldr, *vUoldi,  *vV, *vUnewr, *vUnewi;


    /*GPU variables:*/
    double *Utmpr, *Utmpi;
    double *Uoutr, *Uouti, *ktotr, *ktoti;
    double *Uoldr_gpu, *Uoldi_gpu,  *V_gpu;

    /*Find the dimensions of the input cube*/
    dims      = (int)mxGetNumberOfDimensions(prhs[0]);
    dim_array = (const int*)mxGetDimensions(prhs[0]);
    M         = dim_array[0];
    N         = dim_array[1];
    L         = dim_array[2];

     
    
    /*Create output vector*/
    plhs[0] =  mxCreateNumericArray((mwSize)dims, (mwSize*)dim_array, mxDOUBLE_CLASS, mxCOMPLEX);

    /* Retrieve the input data */
    vUoldr = mxGetPr(prhs[0]);
    if(mxIsComplex(prhs[0])){
        vUoldi = mxGetPi(prhs[0]);
    }
    else{
        vUoldi = (double *)malloc(sizeof(double)*L*N*M);
        for(i=0;i<L*N*M;i++){
            vUoldi[i] = 0.0;
        }
    }
    vV     = mxGetPr(prhs[1]);

    /*Get the rest of the input variables*/
    s           = (double)mxGetScalar(prhs[2]);
    a           = (double)mxGetScalar(prhs[3]);
    h2          = (double)mxGetScalar(prhs[4]);
    BC          =    (int)mxGetScalar(prhs[5]);
    chunk_size  =    (int)mxGetScalar(prhs[6]);
    K           = (double)mxGetScalar(prhs[7]);

    /*Pre-compute parameter divisions*/
    ah2 = a/h2;
    K2  = K/2.0;
    K6  = K/6.0;

    /*Allocate 1D CUDA memory*/
    cudaMalloc((void**) &Uoldr_gpu, M*N*L*sizeof(double));
    cudaMalloc((void**) &Uoldi_gpu, M*N*L*sizeof(double));
    cudaMalloc((void**) &V_gpu,     M*N*L*sizeof(double));
    cudaMalloc((void**) &Utmpr,     M*N*L*sizeof(double));
    cudaMalloc((void**) &Utmpi,     M*N*L*sizeof(double));
    cudaMalloc((void**) &Uouti,     M*N*L*sizeof(double));
    cudaMalloc((void**) &Uoutr,     M*N*L*sizeof(double));
    cudaMalloc((void**) &ktotr,     M*N*L*sizeof(double));
    cudaMalloc((void**) &ktoti,     M*N*L*sizeof(double));


    /*Copy input vectors to GPU*/
    cudaMemcpy(Uoldr_gpu, vUoldr, M*N*L*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Uoldi_gpu, vUoldi, M*N*L*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(V_gpu,     vV,     M*N*L*sizeof(double),cudaMemcpyHostToDevice);

    /*Set up CUDA grid and block size*/
    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY,BLOCK_SIZEZ);

    /*Compute desired y grid dimension*/
    gridDim_y  = (int)ceil((N+0.0)/dimBlock.y);

    /*For 3D need to extend y-grid dimention to include z-cuts:*/
    dim3 dimGrid((int)ceil((M+0.0)/dimBlock.x), gridDim_y*((int)ceil((L+0.0)/dimBlock.z)));

  
    /* 
    printf("N: %d\n",N);
    printf("L: %d\n",L);
    printf("M: %d\n",M);  
    printf("BlockX: %d\n",BLOCK_SIZEX);   
    printf("BlockY: %d\n",BLOCK_SIZEY);   
    printf("BlockZ: %d\n",BLOCK_SIZEZ); 
    printf("gridDim_y: %d\n",gridDim_y);  
    printf("dimGridx: %d\n",(int)ceil((M+0.0)/dimBlock.x));
    printf("dimGridy: %d\n",gridDim_y*((int)ceil((L+0.0)/dimBlock.z)));
    */
      
    
    /*Compute chunk of time steps*/
    for (c=0; c<chunk_size; c++)
    {
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoldr_gpu,Uoldi_gpu,Utmpr,    Utmpi,    V_gpu,V_gpu,V_gpu,s,ah2,BC,L,N,M,gridDim_y,K2,1);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,Uoutr,Uouti,V_gpu,s,ah2,BC,L,N,M,gridDim_y,K2,2);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoutr,    Uouti,    Uoldr_gpu,Uoldi_gpu,Utmpr,Utmpi,V_gpu,s,ah2,BC,L,N,M,gridDim_y,K, 2);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,V_gpu,V_gpu,V_gpu,s,ah2,BC,L,N,M,gridDim_y,K6,3);
    }

    /*Set up output vectors*/
    vUnewr = mxGetPr(plhs[0]);
    vUnewi = mxGetPi(plhs[0]);
    
    /*Make sure everything is done (important for large chunk-size computations)*/
    cudaDeviceSynchronize();

    /*Transfer solution back to CPU*/
    cudaMemcpy(vUnewr,Uoldr_gpu, M*N*L*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vUnewi,Uoldi_gpu, M*N*L*sizeof(double),cudaMemcpyDeviceToHost);

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

    if(!mxIsComplex(prhs[0])){
        free(vUoldi);
    }

    cudaDeviceReset();

}

/*For reference, command to compile code in MATLAB on windows:
nvmex -f nvmexopts.bat NLSE3D_TAKE_STEPS_CD_CUDA_D.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
*/

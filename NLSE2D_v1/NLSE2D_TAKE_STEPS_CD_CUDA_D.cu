/*----------------------------
NLSE2D_TAKE_STEPS_CD_CUDA_D.cu
Program to integrate a chunk of time steps of the 2D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy) - V(x,y)*U + s*|U|^2*U = 0
using RK4 + CD with CUDA compatable GPUs in
double precision (CUDA 1.3 and up).

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
__global__ void compute_F(double* ktotr, double* ktoti,
                          double* Utmpr, double* Utmpi,
                          double* Uoldr, double* Uoldi,
                          double* Uoutr, double* Uouti,
                          double* V,     double s,     double ah2,
                          int BC, int N, int M, int pitch, double k, int fstep)
{
     /*Declare shared memory space*/
     __shared__ double sUtmpr[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
     __shared__ double sUtmpi[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
     __shared__ double  NLSFr[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
     __shared__ double  NLSFi[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
     __shared__ double     sV[BLOCK_SIZEY+2][BLOCK_SIZEX+2];

    /*Create four indexes:  two for shared, two for global*/
    int i   = blockIdx.y*blockDim.y+threadIdx.y;
    int j   = blockIdx.x*blockDim.x+threadIdx.x;
    int si  = threadIdx.y + 1;
    int sj  = threadIdx.x + 1;   

    int msd_si, msd_sj;
    double OM;

    int ij;
    ij = pitch*i+j;

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
        if(!((j>0 && j<M-1)&&(i>0 && i<N-1)))  /* if( ((i==0 || j==0) || (i==N-1 || j==M-1)) && (i<N && j<M)) */
        {
            switch(BC){
                case 1: /*Dirichlet*/
                    NLSFr[si][sj]   = 0.0;
                    NLSFi[si][sj]   = 0.0;
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
                    NLSFr[si][sj]   = 0.0;
                    NLSFi[si][sj]   = 0.0;
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
    double h2, a, ah2, s, k, k2, k6;
    double *vUoldr, *vUoldi,  *vV;
    double *vUnewr, *vUnewi;
    size_t pitch;
    int ipitch;

    /*GPU variables:*/
    double *Uoldr_gpu, *Uoldi_gpu, *V_gpu;
    double *ktotr, *ktoti;
    double *Utmpr,*Utmpi;
    double *Uoutr,*Uouti;

    /*Find the dimensions of the input matrix*/
    N = mxGetN(prhs[0]);
    M = mxGetM(prhs[0]);

    /*Create output vector*/
    plhs[0] = mxCreateDoubleMatrix(M,N,mxCOMPLEX);

    /* Retrieve the input data */
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
    s           = (double)mxGetScalar(prhs[2]);
    a           = (double)mxGetScalar(prhs[3]);
    h2          = (double)mxGetScalar(prhs[4]);
    BC          =    (int)mxGetScalar(prhs[5]);
    chunk_size  =    (int)mxGetScalar(prhs[6]);
    k           = (double)mxGetScalar(prhs[7]);

    /*Pre-compute parameter divisions*/
    ah2 = a/h2;
    k2  = k/2.0;
    k6  = k/6.0;

    /*Allocate 2D CUDA memory*/
    cudaMallocPitch((void**) &Uoldr_gpu, &pitch, M*sizeof(double), N);
    cudaMallocPitch((void**) &Uoldi_gpu, &pitch, M*sizeof(double), N);
    cudaMallocPitch((void**) &V_gpu,     &pitch, M*sizeof(double), N);
    cudaMallocPitch((void**) &Utmpr,     &pitch, M*sizeof(double), N);
    cudaMallocPitch((void**) &Utmpi,     &pitch, M*sizeof(double), N);
    cudaMallocPitch((void**) &Uouti,     &pitch, M*sizeof(double), N);
    cudaMallocPitch((void**) &Uoutr,     &pitch, M*sizeof(double), N);
    cudaMallocPitch((void**) &ktotr,     &pitch, M*sizeof(double), N);
    cudaMallocPitch((void**) &ktoti,     &pitch, M*sizeof(double), N);
  
    /*Copy input vectors to GPU*/
    cudaMemcpy2D(Uoldr_gpu,pitch, vUoldr,M*sizeof(double),M*sizeof(double),N,cudaMemcpyHostToDevice);
    cudaMemcpy2D(Uoldi_gpu,pitch, vUoldi,M*sizeof(double),M*sizeof(double),N,cudaMemcpyHostToDevice);
    cudaMemcpy2D(V_gpu,    pitch, vV,    M*sizeof(double),M*sizeof(double),N,cudaMemcpyHostToDevice);

    /*Compute index for pitch vector memory*/
    ipitch = (int)(pitch/sizeof(double));
        
    /*Set up CUDA grid and block size*/
    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY);
    dim3 dimGrid((int)ceil((M+0.0)/dimBlock.x), (int)ceil((N+0.0)/dimBlock.y));

    /*
    printf("M: %d\n",M);
    printf("N: %d\n",N);
    printf("pitch: %d\n",ipitch);
    printf("dimGridx: %d\n",(int)ceil((M+0.0)/dimBlock.x));
    printf("dimGridy: %d\n",(int)ceil((N+0.0)/dimBlock.y));
    */
    
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
    cudaMemcpy2D(vUnewr,M*sizeof(double),Uoldr_gpu,pitch,M*sizeof(double),N,cudaMemcpyDeviceToHost);
    cudaMemcpy2D(vUnewi,M*sizeof(double),Uoldi_gpu,pitch,M*sizeof(double),N,cudaMemcpyDeviceToHost);

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
    if(!mxIsComplex(prhs[0])){
        free(vUoldi);
    }

    cudaDeviceReset();

}

/*For reference, command to compile code in MATLAB on windows:
nvmex -f nvmexopts.bat NLSE2D_TAKE_STEPS_CD_CUDA_D.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
*/

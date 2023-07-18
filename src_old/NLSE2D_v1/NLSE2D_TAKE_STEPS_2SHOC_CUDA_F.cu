/*----------------------------
NLSE2D_TAKE_STEPS_2SHOC_CUDA_F.cu
Program to integrate a chunk of time steps of the 2D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy) + V(x,y)*U + s*|U|^2*U = 0
using RK4 + 2SHOC with CUDA compatable GPUs in
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
                          float* Dr,    float* Di,
                          float* V,     float s,     float a,
                          float a1_6h2, float a1_12, float h2,
                          int BC,       int N,
                          int M,        int pitch, float k, int fstep)
{
    /*Declare shared memory space*/
    __shared__ float sUtmpr[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float sUtmpi[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float  NLSFr[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float  NLSFi[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float    sDr[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float    sDi[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float     sV[BLOCK_SIZEY+2][BLOCK_SIZEX+2];

    /*Create four indexes:  two for shared, two for global*/

    int j   = blockIdx.x*blockDim.x+threadIdx.x;
    int sj  = threadIdx.x+1;

    int i   = blockIdx.y*blockDim.y+threadIdx.y;
    int si  = threadIdx.y+1;

    int msd_si,msd_sj;
    int ij = pitch*i+j;
    float OM;

    /*Copy vector from global memory into shared memory*/
    if(i<N && j<M)
    {
        sUtmpr[si][sj] = Utmpr[ij];
        sUtmpi[si][sj] = Utmpi[ij];
           sDr[si][sj] =    Dr[ij];
           sDi[si][sj] =    Di[ij];
            sV[si][sj] =     V[ij];

      /*Copy boundary layer of shared memory block*/
        /*These need to be outside next if statement
          due to needed diag accesses of Utmp (D thrown in)*/
        if(i>0 && si==1)
        {
            sUtmpr[0][sj] = Utmpr[ij-pitch];
            sUtmpi[0][sj] = Utmpi[ij-pitch];
            sDr[0][sj]    =    Dr[ij-pitch];
            sDi[0][sj]    =    Di[ij-pitch];
        }
        if(i<N-1 && si==blockDim.y)
        {
            sUtmpr[si+1][sj] = Utmpr[ij+pitch];
            sUtmpi[si+1][sj] = Utmpi[ij+pitch];
            sDr[si+1][sj]    =    Dr[ij+pitch];
            sDi[si+1][sj]    =    Di[ij+pitch];
        }
        if(j>0 && sj==1)
        {
            sUtmpr[si][0] = Utmpr[ij-1];
            sUtmpi[si][0] = Utmpi[ij-1];
            sDr[si][0]    =    Dr[ij-1];
            sDi[si][0]    =    Di[ij-1];
        }
        if(j<M-1 && sj==blockDim.x)
        {
            sUtmpr[si][sj+1] = Utmpr[ij+1];
            sUtmpi[si][sj+1] = Utmpi[ij+1];
            sDr[si][sj+1]    =    Dr[ij+1];
            sDi[si][sj+1]    =    Di[ij+1];
        }
    }/*end*/

    /*Synchronize the threads in the block so that all shared cells are filled.*/
    __syncthreads();

    /*If cell is not on boundary...*/
        if ((j>0 && j<M-1) && (i>0 && i<N-1) )
        {
            /*Copy boundary layer of shared memory block*/
            if(si==1 && sj==1)
            {
                sUtmpr[0][0] = Utmpr[ij-pitch-1];
                sUtmpi[0][0] = Utmpi[ij-pitch-1];
            }
            if(si==1 && sj==blockDim.x)
            {
                sUtmpr[0][sj+1] = Utmpr[ij-pitch+1];
                sUtmpi[0][sj+1] = Utmpi[ij-pitch+1];
            }
            if(si==blockDim.y && sj==1)
            {
                sUtmpr[si+1][0] = Utmpr[ij+pitch-1];
                sUtmpi[si+1][0] = Utmpi[ij+pitch-1];
            }
            if(si==blockDim.y && sj==blockDim.x)
            {
                sUtmpr[si+1][sj+1] = Utmpr[ij+pitch+1];
                sUtmpi[si+1][sj+1] = Utmpi[ij+pitch+1];
            }

            NLSFr[si][sj] = -a*sDi[si][sj] + a1_12*(sDi[si+1][sj] + sDi[si-1][sj] + sDi[si][sj+1] + sDi[si][sj-1])
               - a1_6h2*(sUtmpi[si+1][sj+1] + sUtmpi[si+1][sj-1] - 4*sUtmpi[si][sj] + sUtmpi[si-1][sj+1] + sUtmpi[si-1][sj-1])
               - (s*(sUtmpr[si][sj]*sUtmpr[si][sj] + sUtmpi[si][sj]*sUtmpi[si][sj]) - sV[si][sj])*sUtmpi[si][sj];

            NLSFi[si][sj] =  a*sDr[si][sj] - a1_12*(sDr[si+1][sj] + sDr[si-1][sj] + sDr[si][sj+1] + sDr[si][sj-1])
               + a1_6h2*(sUtmpr[si+1][sj+1] + sUtmpr[si+1][sj-1] - 4*sUtmpr[si][sj] + sUtmpr[si-1][sj+1] + sUtmpr[si-1][sj-1])
               + (s*(sUtmpr[si][sj]*sUtmpr[si][sj] + sUtmpi[si][sj]*sUtmpi[si][sj]) - sV[si][sj])*sUtmpr[si][sj];

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


/*Kernel to evaluate D(Psi) using shared memory*/
__global__ void compute_D(float* Dr,    float* Di,
                          float* Utmpr, float* Utmpi,
                          float* V,     float  s,
                          float l_a,    float lh2,
                          int BC,       int N,
                          int M,        int pitch)
{
    /*Declare shared memory space*/
    __shared__ float sUtmpr[BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float sUtmpi[BLOCK_SIZEY+2][BLOCK_SIZEX+2];

    int j   = blockIdx.x*blockDim.x+threadIdx.x;
    int sj  = threadIdx.x + 1;

    int i   = blockIdx.y*blockDim.y+threadIdx.y;
    int si  = threadIdx.y + 1;

    int msd_ij,msd_si,msd_sj;
    int ij = pitch*i+j;
    float A,Nb,Nb1;

    if(i<N && j<M)
    {
        /*Copy blocksized matrix from global memory into shared memory*/
        sUtmpr[si][sj] = Utmpr[ij];
        sUtmpi[si][sj] = Utmpi[ij];
    }

    /*Synchronize the threads in the block so that all shared cells are filled.*/
    __syncthreads();

    /*If cell is NOT on boundary...*/
    if ((j>0 && j<M-1)&&(i>0 && i<N-1))
    {
        /*Copy boundary layer of shared memory block*/
        if(si==1)
        {
            sUtmpr[si-1][sj] = Utmpr[pitch*(i-1) + j];
            sUtmpi[si-1][sj] = Utmpi[pitch*(i-1) + j];
        }
        if(si==blockDim.y)
        {
            sUtmpr[si+1][sj] = Utmpr[pitch*(i+1) + j];
            sUtmpi[si+1][sj] = Utmpi[pitch*(i+1) + j];
        }
        if(sj==1)
        {
            sUtmpr[si][sj-1] = Utmpr[pitch*i + j-1];
            sUtmpi[si][sj-1] = Utmpi[pitch*i + j-1];
        }
        if(sj==blockDim.x)
        {
            sUtmpr[si][sj+1] = Utmpr[pitch*i + j+1];
            sUtmpi[si][sj+1] = Utmpi[pitch*i + j+1];
        }
        /*No synchthreads needed in this case*/

        Di[ij] = (sUtmpi[si][sj+1] + sUtmpi[si-1][sj] - 4*sUtmpi[si][sj]
                + sUtmpi[si+1][sj] + sUtmpi[si][sj-1])*lh2;

        Dr[ij] = (sUtmpr[si+1][sj] + sUtmpr[si-1][sj] - 4*sUtmpr[si][sj]
                + sUtmpr[si][sj+1] + sUtmpr[si][sj-1])*lh2;

     }/*End of interier points*/

     if(BC==2)   __syncthreads(); /* needed for MSD*/

     if(i<N && j<M)
     {
        /*Cell is ON Boundery*/
        if(!((j>0 && j<M-1)&&(i>0 && i<N-1)))
        {
            switch(BC){
                case 1:  /*Dirichlet*/
                    Dr[ij]   = -l_a*(s*(sUtmpr[si][sj]*sUtmpr[si][sj] + sUtmpi[si][sj]*sUtmpi[si][sj]) - V[ij])*sUtmpr[si][sj];
                    Di[ij]   = -l_a*(s*(sUtmpr[si][sj]*sUtmpr[si][sj] + sUtmpi[si][sj]*sUtmpi[si][sj]) - V[ij])*sUtmpi[si][sj];
                    break;
                case 2: /* Mod-Squared Dirichlet |U|^2=B */
                    if(i==0)                msd_si = si+1;
                    if(i==N-1)              msd_si = si-1;
                    if((i!=0) && (i!=N-1))  msd_si = si;
                    if(j==0)                msd_sj = sj+1;
                    if(j==M-1)              msd_sj = sj-1;
                    if((j!=0) && (j!=M-1))  msd_sj = sj;

                    msd_ij = pitch*(i+(msd_si-si)) + j+(msd_sj-sj);

                    Nb  = s*(sUtmpr[si][sj]*sUtmpr[si][sj] + sUtmpi[si][sj]*sUtmpi[si][sj]) - V[ij];
                    Nb1 = s*(sUtmpr[msd_si][msd_sj]*sUtmpr[msd_si][msd_sj] + sUtmpi[msd_si][msd_sj]*sUtmpi[msd_si][msd_sj]) - V[msd_ij];

                    A = (Dr[msd_ij]*sUtmpr[msd_si][msd_sj] + Di[msd_ij]*sUtmpi[msd_si][msd_sj])/
                        (sUtmpr[msd_si][msd_sj]*sUtmpr[msd_si][msd_sj] + sUtmpi[msd_si][msd_sj]*sUtmpi[msd_si][msd_sj]);
                    
                    Dr[ij]  = (A + l_a*(Nb1-Nb))*sUtmpr[si][sj];
                    Di[ij]  = (A + l_a*(Nb1-Nb))*sUtmpi[si][sj];
                    break;
                case 3: /*Lap=0*/
                    Dr[ij] = 0.0f;
                    Di[ij] = 0.0f;
                    break;
                default:
                    Dr[ij] = 0.0f;
                    Di[ij] = 0.0f;
                    break;
            }/*BC Switch*/
        }/*BC*/
    }/*end*/
}/*Compute D*/





/*Main mex function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int    i, j, N, M, chunk_size, BC;
    float  h2, lh2, a, l_a, a1_6h2, a1_12, s, k, k2, k6;
    double *vUoldr, *vUoldi,  *vV;
    double *vUnewr, *vUnewi;
    float  *fvUoldr, *fvUoldi,  *fvV;
    size_t pitch;
    int ipitch;

    /*GPU variables:*/
    float *Utmpr, *Utmpi;
    float *Uoutr, *Uouti, *ktotr, *ktoti;
    float *Uoldr_gpu, *Uoldi_gpu,  *V_gpu;
    float *Dr, *Di;

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
    s           = (float)mxGetScalar(prhs[2]);
    a           = (float)mxGetScalar(prhs[3]);
    h2          = (float)mxGetScalar(prhs[4]);
    BC          =   (int)mxGetScalar(prhs[5]);
    chunk_size  =   (int)mxGetScalar(prhs[6]);
    k           = (float)mxGetScalar(prhs[7]);

    /*Pre-compute parameter divisions*/
    l_a    = 1.0f/a;
    lh2    = 1.0f/h2;
    k2     = k/2.0f;
    k6     = k/6.0f;
    a1_6h2 = a*(1.0f/(6.0f*h2));
    a1_12  = a*(1.0f/12.0f);

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
    cudaMallocPitch((void**) &Dr,        &pitch, M*sizeof(float), N);
    cudaMallocPitch((void**) &Di,        &pitch, M*sizeof(float), N);
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
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Uoldr_gpu,Uoldi_gpu,V_gpu,s,l_a,lh2,BC,N,M,ipitch);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoldr_gpu,Uoldi_gpu,Utmpr,    Utmpi,    V_gpu,V_gpu,Dr,Di,V_gpu,s,a,a1_6h2,a1_12,h2,BC,N,M,ipitch,k2,1);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Utmpr,Utmpi,        V_gpu,s,l_a,lh2,BC,N,M,ipitch);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,Uoutr,Uouti,Dr,Di,V_gpu,s,a,a1_6h2,a1_12,h2,BC,N,M,ipitch,k2,2);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Uoutr,Uouti,        V_gpu,s,l_a,lh2,BC,N,M,ipitch);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoutr,    Uouti,    Uoldr_gpu,Uoldi_gpu,Utmpr,Utmpi,Dr,Di,V_gpu,s,a,a1_6h2,a1_12,h2,BC,N,M,ipitch,k, 2);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Utmpr,Utmpi,        V_gpu,s,l_a,lh2,BC,N,M,ipitch);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,V_gpu,V_gpu,Dr,Di,V_gpu,s,a,a1_6h2,a1_12,h2,BC,N,M,ipitch,k6,3);
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
    cudaFree(Dr);
    cudaFree(Di);

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
nvmex -f nvmexopts.bat NLSE2D_TAKE_STEPS_2SHOC_CUDA_F.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
*/

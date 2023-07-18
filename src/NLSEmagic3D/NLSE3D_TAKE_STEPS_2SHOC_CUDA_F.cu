/*----------------------------
NLSE3D_TAKE_STEPS_2SHOC_CUDA_F.cu:
Program to integrate a chunk of time steps of the 3D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy + Uzz) - V(r)*U + s*|U|^2*U = 0
using RK4 + 2SHOC with CUDA compatable GPUs in
single precision.

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
const int BLOCK_SIZEX = 12;
const int BLOCK_SIZEY = 12;
const int BLOCK_SIZEZ = 4;

/*Kernel to evaluate D(Psi) using shared memory*/
__global__ void compute_D(float* Dr,    float* Di,
                          float* Utmpr, float* Utmpi,
                          float* V,     float  s,
                          float l_a,    float lh2,
                          int BC,       int L,
                          int N,        int M,  int gridDim_y)
{
    /*Declare shared memory space*/
    __shared__ float sUtmpr[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float sUtmpi[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];

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

    int msd_ijk, msd_si, msd_sj, msd_sk;
    float A, Nb, Nb1;

    int ijk = N*M*i + M*j + k;

    if(i<L && (j<N && k<M)){
        /*Copy blocksized matrix from global memory into shared memory*/
        sUtmpr[si][sj][sk] = Utmpr[ijk];
        sUtmpi[si][sj][sk] = Utmpi[ijk];
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

        Di[ijk] = (sUtmpi[si+1][sj][sk] - 6*sUtmpi[si][sj][sk] + sUtmpi[si-1][sj][sk] +
                   sUtmpi[si][sj+1][sk]                        + sUtmpi[si][sj-1][sk] +
                   sUtmpi[si][sj][sk+1]                        + sUtmpi[si][sj][sk-1])*lh2;

        Dr[ijk] = (sUtmpr[si+1][sj][sk] - 6*sUtmpr[si][sj][sk] + sUtmpr[si-1][sj][sk] +
                   sUtmpr[si][sj+1][sk]                        + sUtmpr[si][sj-1][sk] +
                   sUtmpr[si][sj][sk+1]                        + sUtmpr[si][sj][sk-1])*lh2;

     }/*End of interier points*/

     if(BC==2)   __syncthreads(); /* needed for MSD*/

     if(i<L && (j<N && k<M)){

        /*Cell is ON Boundery*/
        if(!( (j>0 && j<N-1) && ((i>0 && i<L-1) && (k>0 && k<M-1)) ) ){
            switch(BC){
                case 1:  /*Dirichlet*/
                    Dr[ijk]   = l_a*(V[ijk] - s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] + sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]))*sUtmpr[si][sj][sk];
                    Di[ijk]   = l_a*(V[ijk] - s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] + sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]))*sUtmpi[si][sj][sk];
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

                    msd_ijk = N*M*(i+(msd_si-si)) + M*(j+(msd_sj-sj)) + k + (msd_sk-sk);

                    if(sUtmpr[msd_si][msd_sj][msd_sk]==0 && sUtmpi[msd_si][msd_sj][msd_sk]==0)
                    {
                        A=0;
                    }
                    else{ 
                    A   = (Dr[msd_ijk]*sUtmpr[msd_si][msd_sj][msd_sk] + Di[msd_ijk]*sUtmpi[msd_si][msd_sj][msd_sk])/
                          (sUtmpr[msd_si][msd_sj][msd_sk]*sUtmpr[msd_si][msd_sj][msd_sk] + sUtmpi[msd_si][msd_sj][msd_sk]*sUtmpi[msd_si][msd_sj][msd_sk]);       
                    } 
                    Nb  = s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] + sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]) - V[ijk];
                    Nb1 = s*(sUtmpr[msd_si][msd_sj][msd_sk]*sUtmpr[msd_si][msd_sj][msd_sk] + sUtmpi[msd_si][msd_sj][msd_sk]*sUtmpi[msd_si][msd_sj][msd_sk]) - V[msd_ijk];	   
		
                    Dr[ijk]  = (A + l_a*(Nb1-Nb))*sUtmpr[si][sj][sk];
                    Di[ijk]  = (A + l_a*(Nb1-Nb))*sUtmpi[si][sj][sk];  
                    break;
                case 3: /*Lap=0*/
                    Dr[ijk] = 0.0f;
                    Di[ijk] = 0.0f;
                    break;                
                default:
                    Dr[ijk] = 0.0f;
                    Di[ijk] = 0.0f;
                    break;
            }/*BC Switch*/
        }/*BC*/
    }/*end*/
}/*Compute D*/

/*Kernel to evaluate F(Psi) using shared memory*/
__global__ void compute_F(float* ktotr, float* ktoti,
                          float* Utmpr, float* Utmpi,
                          float* Uoldr, float* Uoldi,
                          float* Uoutr, float* Uouti,
                          float* Dr,    float* Di,
                          float* V,     float s,     float a,
                          float a1_6h2, float a1_12, float h2,
                          int BC,       int L,
                          int N,        int M,  int gridDim_y, float K, int fstep)
{
    /*Declare shared memory space*/
    __shared__ float sUtmpr[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float sUtmpi[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float  NLSFr[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float  NLSFi[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float    sDr[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float    sDi[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];
    __shared__ float     sV[BLOCK_SIZEZ+2][BLOCK_SIZEY+2][BLOCK_SIZEX+2];

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
    float OM;

    int ijk = N*M*i + M*j + k;

    /*Copy vector from global memory into shared memory*/
    if(i<L && (j<N && k<M))
    {
        sUtmpr[si][sj][sk] = Utmpr[ijk];
        sUtmpi[si][sj][sk] = Utmpi[ijk];
           sDr[si][sj][sk] =    Dr[ijk];
           sDi[si][sj][sk] =    Di[ijk];
            sV[si][sj][sk] =     V[ijk];

        /*Copy boundary layer of shared memory block*/
        /*These need to be outside next if statement due to diag accesses*/
        if(i>0 && si==1)
        {
            sUtmpr[0][sj][sk] = Utmpr[ijk - N*M];
            sUtmpi[0][sj][sk] = Utmpi[ijk - N*M];
            sDr[0][sj][sk]    =    Dr[ijk - N*M];
            sDi[0][sj][sk]    =    Di[ijk - N*M];
        }
        if(i<L-1 && si==blockDim.z)
        {
            sUtmpr[si+1][sj][sk] = Utmpr[ijk + N*M];
            sUtmpi[si+1][sj][sk] = Utmpi[ijk + N*M];
            sDr[si+1][sj][sk]    =    Dr[ijk + N*M];
            sDi[si+1][sj][sk]    =    Di[ijk + N*M];
        }
        if(j>0 && sj==1)
        {
            sUtmpr[si][0][sk] = Utmpr[ijk - M];
            sUtmpi[si][0][sk] = Utmpi[ijk - M];
            sDr[si][0][sk]    =    Dr[ijk - M];
            sDi[si][0][sk]    =    Di[ijk - M];
        }
        if(j<N-1 && sj==blockDim.y)
        {
            sUtmpr[si][sj+1][sk] = Utmpr[ijk + M];
            sUtmpi[si][sj+1][sk] = Utmpi[ijk + M];
            sDr[si][sj+1][sk]    =    Dr[ijk + M];
            sDi[si][sj+1][sk]    =    Di[ijk + M];
        }
        if(k>0 && sk==1)
        {
            sUtmpr[si][sj][0] = Utmpr[ijk - 1];
            sUtmpi[si][sj][0] = Utmpi[ijk - 1];
            sDr[si][sj][0]    =    Dr[ijk - 1];
            sDi[si][sj][0]    =    Di[ijk - 1];
        }
        if(k<M-1 && sk==blockDim.x)
        {
            sUtmpr[si][sj][sk+1] = Utmpr[ijk + 1];
            sUtmpi[si][sj][sk+1] = Utmpi[ijk + 1];
            sDr[si][sj][sk+1]    =    Dr[ijk + 1];
            sDi[si][sj][sk+1]    =    Di[ijk + 1];
        }
    }/*end*/

    /*Synchronize the threads in the block so that all shared cells are filled.*/
    __syncthreads();

    /*If cell is not on boundary...*/
    if((j>0 && j<N-1) && ((i>0 && i<L-1) && (k>0 && k<M-1)))
    {
        /*Copy boundary layer of shared memory block*/
        if(si==1 && sj==1)
        {
            sUtmpr[0][0][sk] = Utmpr[ijk - N*M - M];
            sUtmpi[0][0][sk] = Utmpi[ijk - N*M - M];
        }
        if(si==1 && sj==blockDim.y)
        {
            sUtmpr[0][sj+1][sk] = Utmpr[ijk - N*M + M];
            sUtmpi[0][sj+1][sk] = Utmpi[ijk - N*M + M];
        }
        if(si==1 && sk==1)
        {
            sUtmpr[0][sj][0] = Utmpr[ijk - N*M - 1];
            sUtmpi[0][sj][0] = Utmpi[ijk - N*M - 1];
        }
        if(si==1 && sk==blockDim.x)
        {
            sUtmpr[0][sj][sk+1] = Utmpr[ijk - N*M + 1];
            sUtmpi[0][sj][sk+1] = Utmpi[ijk - N*M + 1];
        }
        if(si==blockDim.z && sj==1)
        {
            sUtmpr[si+1][0][sk] = Utmpr[ijk + N*M - M];
            sUtmpi[si+1][0][sk] = Utmpi[ijk + N*M - M];
        }
        if(si==blockDim.z && sj==blockDim.y)
        {
            sUtmpr[si+1][sj+1][sk] = Utmpr[ijk + N*M + M];
            sUtmpi[si+1][sj+1][sk] = Utmpi[ijk + N*M + M];
        }
        if(si==blockDim.z && sk==1)
        {
            sUtmpr[si+1][sj][0] = Utmpr[ijk + N*M - 1];
            sUtmpi[si+1][sj][0] = Utmpi[ijk + N*M - 1];
        }
        if(si==blockDim.z && sk==blockDim.x)
        {
            sUtmpr[si+1][sj][sk+1] = Utmpr[ijk + N*M + 1];
            sUtmpi[si+1][sj][sk+1] = Utmpi[ijk + N*M + 1];
        }
        if(sj==1 && sk==1)
        {
            sUtmpr[si][0][0] = Utmpr[ijk - M - 1];
            sUtmpi[si][0][0] = Utmpi[ijk - M - 1];
        }
        if(sj==1 && sk==blockDim.x)
        {
            sUtmpr[si][0][sk+1] = Utmpr[ijk - M + 1];
            sUtmpi[si][0][sk+1] = Utmpi[ijk - M + 1];
        }
        if(sj==blockDim.y && sk==1)
        {
            sUtmpr[si][sj+1][0] = Utmpr[ijk + M - 1];
            sUtmpi[si][sj+1][0] = Utmpi[ijk + M - 1];
        }
        if(sj==blockDim.y && sk==blockDim.x)
        {
            sUtmpr[si][sj+1][sk+1] = Utmpr[ijk + M + 1];
            sUtmpi[si][sj+1][sk+1] = Utmpi[ijk + M + 1];
        }


        NLSFr[si][sj][sk]     =  a1_12*(sDi[si+1][sj][sk] + sDi[si-1][sj][sk] +
                                sDi[si][sj+1][sk] + sDi[si][sj-1][sk] +
                                sDi[si][sj][sk+1] + sDi[si][sj][sk-1] -
                             10*sDi[si][sj][sk])
         - a1_6h2*(sUtmpi[si+1][sj+1][sk] + sUtmpi[si][sj+1][sk+1] + sUtmpi[si+1][sj][sk+1] +
                   sUtmpi[si-1][sj-1][sk] + sUtmpi[si][sj-1][sk-1] + sUtmpi[si-1][sj][sk-1] +
                   sUtmpi[si+1][sj-1][sk] + sUtmpi[si][sj+1][sk-1] + sUtmpi[si+1][sj][sk-1] +
                   sUtmpi[si-1][sj+1][sk] + sUtmpi[si][sj-1][sk+1] + sUtmpi[si-1][sj][sk+1] -
                12*sUtmpi[si][sj][sk])
         + (sV[si][sj][sk] - s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] + sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]))*sUtmpi[si][sj][sk];

        NLSFi[si][sj][sk]    = -a1_12*(sDr[si+1][sj][sk] + sDr[si-1][sj][sk] +
                               sDr[si][sj+1][sk] + sDr[si][sj-1][sk] +
                               sDr[si][sj][sk+1] + sDr[si][sj][sk-1] -
                            10*sDr[si][sj][sk])
         + a1_6h2*(sUtmpr[si+1][sj+1][sk] + sUtmpr[si][sj+1][sk+1] + sUtmpr[si+1][sj][sk+1] +
                   sUtmpr[si-1][sj-1][sk] + sUtmpr[si][sj-1][sk-1] + sUtmpr[si-1][sj][sk-1] +
                   sUtmpr[si+1][sj-1][sk] + sUtmpr[si][sj+1][sk-1] + sUtmpr[si+1][sj][sk-1] +
                   sUtmpr[si-1][sj+1][sk] + sUtmpr[si][sj-1][sk+1] + sUtmpr[si-1][sj][sk+1] -
                12*sUtmpr[si][sj][sk])
         - (sV[si][sj][sk] - s*(sUtmpr[si][sj][sk]*sUtmpr[si][sj][sk] + sUtmpi[si][sj][sk]*sUtmpi[si][sj][sk]))*sUtmpr[si][sj][sk];

    }/*End of interier points*/


    if(BC==2)   __syncthreads(); /* needed for MSD*/

    if(i<L && (j<N && k<M)){

        /*Cell is ON Boundery*/
        if(!( (j>0 && j<N-1) && ((i>0 && i<L-1) && (k>0 && k<M-1)) ) ){
            switch(BC){
                case 1: /*Dirichlet*/
                    NLSFr[si][sj][sk]   = 0.0f;
                    NLSFi[si][sj][sk]   = 0.0f;
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
                    NLSFr[si][sj][sk]   = 0.0f;
                    NLSFi[si][sj][sk]   = 0.0f;
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
    float  h2, lh2, a, l_a, a1_6h2, a1_12, s, K, K2, K6;
    double *vUoldr, *vUoldi,  *vV, *vUnewr, *vUnewi;
    float  *fvUoldr, *fvUoldi,  *fvV;

    /*GPU variables:*/
    float *Utmpr, *Utmpi;
    float *Uoutr, *Uouti, *ktotr, *ktoti;
    float *Uoldr_gpu, *Uoldi_gpu,  *V_gpu;
    float *Dr, *Di;

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
    s           = (float)mxGetScalar(prhs[2]);
    a           = (float)mxGetScalar(prhs[3]);
    h2          = (float)mxGetScalar(prhs[4]);
    BC          =   (int)mxGetScalar(prhs[5]);
    chunk_size  =   (int)mxGetScalar(prhs[6]);
    K           = (float)mxGetScalar(prhs[7]);

    /*Pre-compute parameter divisions*/
    l_a    = 1.0f/a;
    lh2    = 1.0f/h2;
    K2     = K/2.0f;
    K6     = K/6.0f;
    a1_6h2 = a*(1.0f/(6.0f*h2));
    a1_12  = a*(1.0f/12.0f);

    /*Allocate float input vectors*/
    fvV     = (float*)malloc(sizeof(float)*L*N*M);
    fvUoldr = (float*)malloc(sizeof(float)*L*N*M);
    fvUoldi = (float*)malloc(sizeof(float)*L*N*M);

    /*Allocate 1D CUDA memory*/
    cudaMalloc((void**) &Uoldr_gpu, M*N*L*sizeof(float));
    cudaMalloc((void**) &Uoldi_gpu, M*N*L*sizeof(float));
    cudaMalloc((void**) &V_gpu,     M*N*L*sizeof(float));
    cudaMalloc((void**) &Utmpr,     M*N*L*sizeof(float));
    cudaMalloc((void**) &Utmpi,     M*N*L*sizeof(float));
    cudaMalloc((void**) &Dr,     M*N*L*sizeof(float));
    cudaMalloc((void**) &Di,     M*N*L*sizeof(float));
    cudaMalloc((void**) &Uouti,     M*N*L*sizeof(float));
    cudaMalloc((void**) &Uoutr,     M*N*L*sizeof(float));
    cudaMalloc((void**) &ktotr,     M*N*L*sizeof(float));
    cudaMalloc((void**) &ktoti,     M*N*L*sizeof(float));

        /*Convert double input vectors to float*/
    for(i=0;i<L*N*M;i++){
      fvV[i]     = (float)vV[i];
      fvUoldr[i] = (float)vUoldr[i];
      fvUoldi[i] = (float)vUoldi[i];
    }

    /*Copy input vectors to GPU*/
    cudaMemcpy(Uoldr_gpu, fvUoldr, M*N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(Uoldi_gpu, fvUoldi, M*N*L*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(V_gpu,     fvV,     M*N*L*sizeof(float),cudaMemcpyHostToDevice);

    /*Set up CUDA grid and block size*/
    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY,BLOCK_SIZEZ);

    /*Compute desired y grid dimension*/
    gridDim_y  = (int)ceil((N+0.0f)/dimBlock.y);

    /*For 3D need to extend y-grid dimention to include z-cuts:*/
    dim3 dimGrid((int)ceil((M+0.0)/dimBlock.x), gridDim_y*((int)ceil((L+0.0)/dimBlock.z)));

    /*Compute chunk of time steps*/
    for (c=0; c<chunk_size; c++)
    {
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Uoldr_gpu,Uoldi_gpu,V_gpu,s,l_a,lh2,BC,L,N,M,gridDim_y);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoldr_gpu,Uoldi_gpu,Utmpr,    Utmpi,    V_gpu,V_gpu,Dr,Di,V_gpu,s,a,a1_6h2,a1_12,h2,BC,L,N,M,gridDim_y,K2,1);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Utmpr,Utmpi,        V_gpu,s,l_a,lh2,BC,L,N,M,gridDim_y);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,Uoutr,Uouti,Dr,Di,V_gpu,s,a,a1_6h2,a1_12,h2,BC,L,N,M,gridDim_y,K2,2);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Uoutr,Uouti,        V_gpu,s,l_a,lh2,BC,L,N,M,gridDim_y);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Uoutr,    Uouti,    Uoldr_gpu,Uoldi_gpu,Utmpr,Utmpi,Dr,Di,V_gpu,s,a,a1_6h2,a1_12,h2,BC,L,N,M,gridDim_y,K, 2);
      compute_D <<<dimGrid,dimBlock>>>(Dr,Di,Utmpr,Utmpi,        V_gpu,s,l_a,lh2,BC,L,N,M,gridDim_y);
      compute_F <<<dimGrid,dimBlock>>>(ktotr,ktoti,Utmpr,    Utmpi,    Uoldr_gpu,Uoldi_gpu,V_gpu,V_gpu,Dr,Di,V_gpu,s,a,a1_6h2,a1_12,h2,BC,L,N,M,gridDim_y,K6,3);
    }

    /*Set up output vectors*/
    vUnewr = mxGetPr(plhs[0]);
    vUnewi = mxGetPi(plhs[0]);
    
    /*Make sure everything is done (important for large chunk-size computations)*/
    cudaDeviceSynchronize();

    /*Transfer solution back to CPU*/
    cudaMemcpy(fvUoldr,Uoldr_gpu, M*N*L*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(fvUoldi,Uoldi_gpu, M*N*L*sizeof(float),cudaMemcpyDeviceToHost);

    /*Convert float vector to double output vector*/
    for(i=0;i<L*N*M;i++){
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
nvmex -f nvmexopts.bat NLSE3D_TAKE_STEPS_2SHOC_CUDA_F.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
*/

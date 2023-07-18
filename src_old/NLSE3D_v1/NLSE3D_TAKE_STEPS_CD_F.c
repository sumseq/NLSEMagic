/*----------------------------
NLSE3D_TAKE_STEPS_CD_F.c
Program to integrate a chunk of time steps of the 3D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy + Uzz) - V(r)*U + s*|U|^2*U = 0
using RK4 + CD in single precision.

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
#include "mex.h"
#include "math.h"

void add_cube(float*** Ar,    float*** Ai,
              float*** Br,    float*** Bi,
              float*** Cr,    float*** Ci,
              float K, int L, int N, int M)
{
    int i,j,k;
    for(i=0;i<L;i++){
       for(j=0;j<N;j++){
           for(k=0;k<M;k++){
              Ar[i][j][k]  = Br[i][j][k] + K*Cr[i][j][k];
              Ai[i][j][k]  = Bi[i][j][k] + K*Ci[i][j][k];
           }
       }
    }
}

void NLSE3D_STEP(float*** NLSFr, float*** NLSFi,
                 float*** Utmpr, float*** Utmpi,
                 float*** V, float s, float ah2,
                 int BC, int L, int N, int M){

int i,j,k;
float OM;
int i1,j1,k1;

for(i=1;i<L-1;i++)
{
  for(j=1;j<N-1;j++)
  {
      for(k=1;k<M-1;k++)
      {
    NLSFr[i][j][k] = -ah2*(Utmpi[i+1][j][k] + Utmpi[i-1][j][k] +
                           Utmpi[i][j+1][k] + Utmpi[i][j-1][k] +
                           Utmpi[i][j][k+1] + Utmpi[i][j][k-1]
                                          - 6*Utmpi[i][j][k])
        + (V[i][j][k] - s*(Utmpr[i][j][k]*Utmpr[i][j][k] +
                           Utmpi[i][j][k]*Utmpi[i][j][k])  )*Utmpi[i][j][k];

    NLSFi[i][j][k] =  ah2*(Utmpr[i+1][j][k] + Utmpr[i-1][j][k] +
                           Utmpr[i][j+1][k] + Utmpr[i][j-1][k] +
                           Utmpr[i][j][k+1] + Utmpr[i][j][k-1]
                                          - 6*Utmpr[i][j][k])
        - (V[i][j][k] - s*(Utmpr[i][j][k]*Utmpr[i][j][k] +
                           Utmpi[i][j][k]*Utmpi[i][j][k])  )*Utmpr[i][j][k];
      }
  }
}
/*Boundary conditions:*/
switch (BC){
    case 1:
    for(i=0;i<L;i++){
        for(j=0;j<N;j++){
           NLSFr[i][j][0]     = 0.0f;
           NLSFi[i][j][0]     = 0.0f;
           NLSFr[i][j][M-1]   = 0.0f;
           NLSFi[i][j][M-1]   = 0.0f;
        }
    }
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
           NLSFr[i][0][k]     = 0.0f;
           NLSFi[i][0][k]     = 0.0f;
           NLSFr[i][N-1][k]   = 0.0f;
           NLSFi[i][N-1][k]   = 0.0f;
        }
    }
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){
           NLSFr[0][j][k]     = 0.0f;
           NLSFi[0][j][k]     = 0.0f;
           NLSFr[L-1][j][k]   = 0.0f;
           NLSFi[L-1][j][k]   = 0.0f;
        }
    }
    break;
    case 2: /*MSD*/
    /*First do L-N planes for k=0 and k=M-1*/
    for(i=0;i<L;i++){
        for(j=0;j<N;j++){
        /*Calculate edge and corner indices*/
        i1 = i;
        j1 = j;
        if(i==0)     i1 = 1;
        if(i==L-1)   i1 = L-2;
        if(j==0)     j1 = 1;
        if(j==N-1)   j1 = N-2;

        OM = (NLSFi[i1][j1][1]*Utmpr[i1][j1][1] - NLSFr[i1][j1][1]*Utmpi[i1][j1][1])/
        (Utmpr[i1][j1][1]*Utmpr[i1][j1][1] + Utmpi[i1][j1][1]*Utmpi[i1][j1][1]);
                                        
        NLSFr[i][j][0]  = -OM*Utmpi[i][j][0];
        NLSFi[i][j][0]  =  OM*Utmpr[i][j][0];  

        OM = (NLSFi[i1][j1][M-2]*Utmpr[i1][j1][M-2] - NLSFr[i1][j1][M-2]*Utmpi[i1][j1][M-2])/
        (Utmpr[i1][j1][M-2]*Utmpr[i1][j1][M-2] + Utmpi[i1][j1][M-2]*Utmpi[i1][j1][M-2]);
                                        
        NLSFr[i][j][M-1]  = -OM*Utmpi[i][j][M-1];
        NLSFi[i][j][M-1]  =  OM*Utmpr[i][j][M-1];  
        }/*j*/
    }/*i*/
    /*Now do L-M planes for j=0 and j=N-1
      not repeating edges and corners above*/
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
        i1 = i;
        if(i==0)     i1 = 1;
        if(i==L-1)   i1 = L-2;

        OM = (NLSFi[i1][1][k]*Utmpr[i1][1][k] - NLSFr[i1][1][k]*Utmpi[i1][1][k])/
        (Utmpr[i1][1][k]*Utmpr[i1][1][k] + Utmpi[i1][1][k]*Utmpi[i1][1][k]);
                                        
        NLSFr[i][0][k]  = -OM*Utmpi[i][0][k];
        NLSFi[i][0][k]  =  OM*Utmpr[i][0][k];  

        OM = (NLSFi[i1][N-2][k]*Utmpr[i1][N-2][k] - NLSFr[i1][N-2][k]*Utmpi[i1][N-2][k])/
        (Utmpr[i1][N-2][k]*Utmpr[i1][N-2][k] + Utmpi[i1][N-2][k]*Utmpi[i1][N-2][k]);
                                        
        NLSFr[i][N-1][k]  = -OM*Utmpi[i][N-1][k];
        NLSFi[i][N-1][k]  =  OM*Utmpr[i][N-1][k];  
        }/*k*/
    }/*i*/
    /*Now do N-M planes for i=0 and i=L-1
      no more edges or corners to worry about*/
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){

        OM = (NLSFi[1][j][k]*Utmpr[1][j][k] - NLSFr[1][j][k]*Utmpi[1][j][k])/
        (Utmpr[1][j][k]*Utmpr[1][j][k] + Utmpi[1][j][k]*Utmpi[1][j][k]);
                                        
        NLSFr[0][j][k]  = -OM*Utmpi[0][j][k];
        NLSFi[0][j][k]  =  OM*Utmpr[0][j][k];  

        OM = (NLSFi[L-2][j][k]*Utmpr[L-2][j][k] - NLSFr[L-2][j][k]*Utmpi[L-2][j][k])/
        (Utmpr[L-2][j][k]*Utmpr[L-2][j][k] + Utmpi[L-2][j][k]*Utmpi[L-2][j][k]);
                                        
        NLSFr[L-1][j][k]  = -OM*Utmpi[L-1][j][k];
        NLSFi[L-1][j][k]  =  OM*Utmpr[L-1][j][k]; 
        }
    }
    break;
    case 3: /*Lap=0*/
    for(i=0;i<L;i++){
        for(j=0;j<N;j++){

           NLSFr[i][j][0]     = -(s*(Utmpr[i][j][0]*Utmpr[i][j][0] + Utmpi[i][j][0]*Utmpi[i][j][0]) - V[i][j][0])*Utmpi[i][j][0];
           NLSFi[i][j][0]     =  (s*(Utmpr[i][j][0]*Utmpr[i][j][0] + Utmpi[i][j][0]*Utmpi[i][j][0]) - V[i][j][0])*Utmpr[i][j][0];
           NLSFr[i][j][M-1]   = -(s*(Utmpr[i][j][M-1]*Utmpr[i][j][M-1] + Utmpi[i][j][M-1]*Utmpi[i][j][M-1]) - V[i][j][M-1])*Utmpi[i][j][M-1];
           NLSFi[i][j][M-1]   =  (s*(Utmpr[i][j][M-1]*Utmpr[i][j][M-1] + Utmpi[i][j][M-1]*Utmpi[i][j][M-1]) - V[i][j][M-1])*Utmpr[i][j][M-1];
        }
    }
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
           NLSFr[i][0][k]     = -(s*(Utmpr[i][0][k]*Utmpr[i][0][k] + Utmpi[i][0][k]*Utmpi[i][0][k]) - V[i][0][k])*Utmpi[i][0][k];
           NLSFi[i][0][k]     =  (s*(Utmpr[i][0][k]*Utmpr[i][0][k] + Utmpi[i][0][k]*Utmpi[i][0][k]) - V[i][0][k])*Utmpr[i][0][k];
           NLSFr[i][N-1][k]   = -(s*(Utmpr[i][N-1][k]*Utmpr[i][N-1][k] + Utmpi[i][N-1][k]*Utmpi[i][N-1][k]) - V[i][N-1][k])*Utmpi[i][N-1][k];
           NLSFi[i][N-1][k]   =  (s*(Utmpr[i][N-1][k]*Utmpr[i][N-1][k] + Utmpi[i][N-1][k]*Utmpi[i][N-1][k]) - V[i][N-1][k])*Utmpr[i][N-1][k];
        }
    }
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){
           NLSFr[0][j][k]     = -(s*(Utmpr[0][j][k]*Utmpr[0][j][k] + Utmpi[0][j][k]*Utmpi[0][j][k]) - V[0][j][k])*Utmpi[0][j][k];
           NLSFi[0][j][k]     =  (s*(Utmpr[0][j][k]*Utmpr[0][j][k] + Utmpi[0][j][k]*Utmpi[0][j][k]) - V[0][j][k])*Utmpr[0][j][k];
           NLSFr[L-1][j][k]   = -(s*(Utmpr[L-1][j][k]*Utmpr[L-1][j][k] + Utmpi[L-1][j][k]*Utmpi[L-1][j][k]) - V[L-1][j][k])*Utmpi[L-1][j][k];
           NLSFi[L-1][j][k]   =  (s*(Utmpr[L-1][j][k]*Utmpr[L-1][j][k] + Utmpi[L-1][j][k]*Utmpi[L-1][j][k]) - V[L-1][j][k])*Utmpr[L-1][j][k];
        }
    }
    break;
    default:
    for(i=0;i<L;i++){
        for(j=0;j<N;j++){
           NLSFr[i][j][0]     = 0.0f;
           NLSFi[i][j][0]     = 0.0f;
           NLSFr[i][j][M-1]   = 0.0f;
           NLSFi[i][j][M-1]   = 0.0f;
        }
    }
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
           NLSFr[i][0][k]     = 0.0f;
           NLSFi[i][0][k]     = 0.0f;
           NLSFr[i][N-1][k]   = 0.0f;
           NLSFi[i][N-1][k]   = 0.0f;
        }
    }
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){
           NLSFr[0][j][k]     = 0.0f;
           NLSFi[0][j][k]     = 0.0f;
           NLSFr[L-1][j][k]   = 0.0f;
           NLSFi[L-1][j][k]   = 0.0f;
        }
    }
    break;
}/*BC Switch*/

}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{

/*Input: U,V,s,a,h2,BC,chunk_size,k*/

mwSize L,N,M,dims;
mwSize *dim_array;
int    i, j, k, c, chunk_size, BC;
float h2, a, s, K, K2, K6, ah2;
double *vUoldr, *vUoldi,  *vUnewr, *vUnewi, *vV;
float *fvUoldr, *fvUoldi, *fvV;
float ***Uoldr, ***Uoldi, ***V;
float ***Utmpr, ***Utmpi;
float ***ktmpr, ***ktmpi, ***ktotr, ***ktoti;

/* Find the dimensions of the vector */
dims = mxGetNumberOfDimensions(prhs[0]);
dim_array = mxGetDimensions(prhs[0]);
M = dim_array[0];
N = dim_array[1];
L = dim_array[2];

/*mexPrintf("L: %d, N: %d, M: %d dims: %d\n",L,N,M,dims);*/

/* Retrieve the input data */
vUoldr = mxGetPr(prhs[0]);
/*If init condition real, need to create imag aray*/
if(mxIsComplex(prhs[0])){
    vUoldi = mxGetPi(prhs[0]);
}
else{
    /*printf("Setting imag to 0---this should only happen once");*/
    vUoldi = (double *)malloc(sizeof(double)*L*N*M);
    for(i=0;i<L*N*M;i++){
        vUoldi[i] = 0.0;
    }
}
vV      = mxGetPr(prhs[1]);

fvV     = malloc(sizeof(float)*L*N*M);
fvUoldr = malloc(sizeof(float)*L*N*M);
fvUoldi = malloc(sizeof(float)*L*N*M);

for(i=0;i<L*N*M;i++){
      fvV[i]     = (float)vV[i];
      fvUoldr[i] = (float)vUoldr[i];
      fvUoldi[i] = (float)vUoldi[i];
}

Uoldr   = (float ***) malloc(sizeof(float**)*L);
Uoldi   = (float ***) malloc(sizeof(float**)*L);
V       = (float ***) malloc(sizeof(float**)*L);
Utmpr   = (float ***) malloc(sizeof(float**)*L);
Utmpi   = (float ***) malloc(sizeof(float**)*L);
ktmpr   = (float ***) malloc(sizeof(float**)*L);
ktmpi   = (float ***) malloc(sizeof(float**)*L);
ktotr   = (float ***) malloc(sizeof(float**)*L);
ktoti   = (float ***) malloc(sizeof(float**)*L);

for (i=0;i<L;i++){
    V[i]     = (float **) malloc(sizeof(float*)*N);
    Uoldr[i] = (float **) malloc(sizeof(float*)*N);
    Uoldi[i] = (float **) malloc(sizeof(float*)*N);
    Utmpr[i] = (float **) malloc(sizeof(float*)*N);
    Utmpi[i] = (float **) malloc(sizeof(float*)*N);
    ktmpr[i] = (float **) malloc(sizeof(float*)*N);
    ktmpi[i] = (float **) malloc(sizeof(float*)*N);
    ktotr[i] = (float **) malloc(sizeof(float*)*N);
    ktoti[i] = (float **) malloc(sizeof(float*)*N);
    for (j=0; j<N; j++){
         V[i][j]     = fvV     + N*M*i + M*j;
         Uoldr[i][j] = fvUoldr + N*M*i + M*j;
         Uoldi[i][j] = fvUoldi + N*M*i + M*j;
         Utmpr[i][j] = (float *) malloc(sizeof(float)*M);
         Utmpi[i][j] = (float *) malloc(sizeof(float)*M);
         ktmpr[i][j] = (float *) malloc(sizeof(float)*M);
         ktmpi[i][j] = (float *) malloc(sizeof(float)*M);
         ktotr[i][j] = (float *) malloc(sizeof(float)*M);
         ktoti[i][j] = (float *) malloc(sizeof(float)*M);
    }
}


s          = (float)mxGetScalar(prhs[2]);
a          = (float)mxGetScalar(prhs[3]);
h2         = (float)mxGetScalar(prhs[4]);
BC         = (int)mxGetScalar(prhs[5]);
chunk_size = (int)mxGetScalar(prhs[6]);
K          = (float)mxGetScalar(prhs[7]);

ah2 = a/h2;
K2  = K/2.0f;
K6  = K/6.0f;

for (c = 0; c<chunk_size; c++)
{
    NLSE3D_STEP(ktotr, ktoti,Uoldr,Uoldi,V,s,ah2,BC,L,N,M);
    add_cube(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,K2,L,N,M);
    NLSE3D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,L,N,M);
    add_cube(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,L,N,M);
    add_cube(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,K2,L,N,M);
    NLSE3D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,L,N,M);
    add_cube(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,L,N,M);
    add_cube(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,K,L,N,M);
    NLSE3D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,L,N,M);
    add_cube(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0f,L,N,M);
    add_cube(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,K6,L,N,M);

}/*End of c chunk_size*/

/* Create an mxArray for the output data */
plhs[0] = mxCreateNumericArray(dims, dim_array, mxDOUBLE_CLASS, mxCOMPLEX);

vUnewr  = mxGetPr(plhs[0]);
vUnewi  = mxGetPi(plhs[0]);

/*Copy 2D array into result vector*/
for(i=0;i<L;i++){
    for(j=0;j<N;j++){
        for (k=0; k<M; k++){
           vUnewr[N*M*i + M*j + k] = (double)Uoldr[i][j][k];
           vUnewi[N*M*i + M*j + k] = (double)Uoldi[i][j][k];
        }
    }
}


/*Free up memory:*/
for (i=0;i<L;i++){
    for(j=0;j<N;j++){
       free(Utmpr[i][j]);
       free(Utmpi[i][j]);
       free(ktmpr[i][j]);
       free(ktmpi[i][j]);
       free(ktotr[i][j]);
       free(ktoti[i][j]);
    }
}
for (i=0; i<L; i++){
       free(Utmpr[i]);
       free(Utmpi[i]);
       free(ktmpr[i]);
       free(ktmpi[i]);
       free(ktotr[i]);
       free(ktoti[i]);
       free(Uoldr[i]);
       free(Uoldi[i]);
       free(V[i]);
}

free(Uoldr);  free(Uoldi);
free(Utmpr);  free(Utmpi);
free(ktmpr);  free(ktmpi);
free(ktotr);  free(ktoti);
free(V);

free(fvV);
free(fvUoldr);
free(fvUoldi);

if(!mxIsComplex(prhs[0])){
    free(vUoldi);
}

}

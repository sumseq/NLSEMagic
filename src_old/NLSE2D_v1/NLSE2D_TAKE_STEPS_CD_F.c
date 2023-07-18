/*----------------------------
NLSE2D_TAKE_STEPS_CD_F.c
Program to integrate a chunk of time steps of the 2D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy) - V(r)*U + s*|U|^2*U = 0
using RK4 + CD in single precision.

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
#include "mex.h"
#include "math.h"

void add_matrix(float** Ar,    float** Ai,
                float** Br,    float** Bi,
                float** Cr,    float** Ci,
                float  k,     int N,  int M)
{
	int i,j;
	for(i=0;i<N;i++){
	   for(j=0;j<M;j++){
    	Ar[i][j]  = Br[i][j] + k*Cr[i][j];  
    	Ai[i][j]  = Bi[i][j] + k*Ci[i][j];  
	   }
    }
}

void NLSE2D_STEP(float** NLSFr, float** NLSFi, 
                 float** Utmpr, float** Utmpi, 
                 float** V, float s, float ah2, 
                 int BC, int N, int M){

int i,j,k,i1,j1;	
float OM;
int msd_x0[4] = {0,0,  N-1,N-1};
int msd_y0[4] = {0,M-1,0,  M-1};
int msd_x1[4] = {1,1,  N-2,N-2};
int msd_y1[4] = {1,M-2,1,  M-2};

for (i=1;i<N-1;i++)
{       
  for(j=1;j<M-1;j++)
  {
    NLSFr[i][j] = -ah2*(Utmpi[i+1][j] - 4*Utmpi[i][j] + Utmpi[i-1][j] + Utmpi[i][j+1] + Utmpi[i][j-1]) 
        - (s*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]) - V[i][j])*Utmpi[i][j];
    
    NLSFi[i][j] =  ah2*(Utmpr[i+1][j] - 4*Utmpr[i][j] + Utmpr[i-1][j] + Utmpr[i][j+1] + Utmpr[i][j-1])
        + (s*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]) - V[i][j])*Utmpr[i][j];
  }    
}
/*Boundary conditions:*/
switch (BC){
    case 1:
    for(i=0;i<N;i++){
        NLSFr[i][0]     = 0.0f; 
        NLSFi[i][0]     = 0.0f;
        NLSFr[i][M-1]   = 0.0f; 
        NLSFi[i][M-1]   = 0.0f;
    }
    for(j=0;j<M;j++){
        NLSFr[0][j]     = 0.0f; 
        NLSFi[0][j]     = 0.0f;
        NLSFr[N-1][j]   = 0.0f; 
        NLSFi[N-1][j]   = 0.0f; 	                  
    }
    break;
    case 2: /*MSD*/
    /*i loop*/
    for(i=1;i<N-1;i++){       

       OM =  (NLSFi[i][1]*Utmpr[i][1] -  NLSFr[i][1]*Utmpi[i][1])/
             (Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1]);
                                        
       NLSFr[i][0]  = -OM*Utmpi[i][0];
       NLSFi[i][0]  =  OM*Utmpr[i][0];

       OM =  (NLSFi[i][M-2]*Utmpr[i][M-2] -  NLSFr[i][M-2]*Utmpi[i][M-2])/
             (Utmpr[i][M-2]*Utmpr[i][M-2] + Utmpi[i][M-2]*Utmpi[i][M-2]);
                                        
       NLSFr[i][M-1]  = -OM*Utmpi[i][M-1];
       NLSFi[i][M-1]  =  OM*Utmpr[i][M-1];
        
    }
    /*j loop*/
    for(j=1;j<M-1;j++){
        
       OM =  (NLSFi[1][j]*Utmpr[1][j] -  NLSFr[1][j]*Utmpi[1][j])/
             (Utmpr[1][j]*Utmpr[1][j] + Utmpi[1][j]*Utmpi[1][j]);
                                        
       NLSFr[0][j]  = -OM*Utmpi[0][j];
       NLSFi[0][j]  =  OM*Utmpr[0][j];

       OM =  (NLSFi[N-2][j]*Utmpr[N-2][j] -  NLSFr[N-2][j]*Utmpi[N-2][j])/
             (Utmpr[N-2][j]*Utmpr[N-2][j] + Utmpi[N-2][j]*Utmpi[N-2][j]);
                                        
       NLSFr[N-1][j]  = -OM*Utmpi[N-1][j];
       NLSFi[N-1][j]  =  OM*Utmpr[N-1][j];        
        
    }
    /*Now do 4 corners*/
    for(k=0;k<4;k++){          
        i  = msd_x0[k];
        j  = msd_y0[k];
        i1 = msd_x1[k];
        j1 = msd_y1[k];   

        OM = (NLSFi[i1][j1]*Utmpr[i1][j1] -  NLSFr[i1][j1]*Utmpi[i1][j1])/
             (Utmpr[i1][j1]*Utmpr[i1][j1] + Utmpi[i1][j1]*Utmpi[i1][j1]);
                                        
        NLSFr[i][j]  = -OM*Utmpi[i][j];
        NLSFi[i][j]  =  OM*Utmpr[i][j];
    }    
    break;
    case 3:        /*Uxx + Uyy = 0: */
	for(i=0;i<N;i++){
       NLSFr[i][0]   = - (s*(Utmpr[i][0]*Utmpr[i][0] + Utmpi[i][0]*Utmpi[i][0]) - V[i][0])*Utmpi[i][0];
	   NLSFi[i][0]   =   (s*(Utmpr[i][0]*Utmpr[i][0] + Utmpi[i][0]*Utmpi[i][0]) - V[i][0])*Utmpr[i][0];
       NLSFr[i][M-1] = - (s*(Utmpr[i][M-1]*Utmpr[i][M-1] + Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1])*Utmpi[i][M-1];
	   NLSFi[i][M-1] =   (s*(Utmpr[i][M-1]*Utmpr[i][M-1] + Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1])*Utmpr[i][M-1];
    }    
    for(j=0;j<M;j++){
       NLSFr[0][j]   = - (s*(Utmpr[0][j]*Utmpr[0][j] + Utmpi[0][j]*Utmpi[0][j]) - V[0][j])*Utmpi[0][j];
	   NLSFi[0][j]   =   (s*(Utmpr[0][j]*Utmpr[0][j] + Utmpi[0][j]*Utmpi[0][j]) - V[0][j])*Utmpr[0][j];
       NLSFr[N-1][j] = - (s*(Utmpr[N-1][j]*Utmpr[N-1][j] + Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j])*Utmpi[N-1][j];
	   NLSFi[N-1][j] =   (s*(Utmpr[N-1][j]*Utmpr[N-1][j] + Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j])*Utmpr[N-1][j];
    }
    break;
    default:
    for(i=0;i<N;i++){
        NLSFr[i][0]     = 0.0f; 
        NLSFi[i][0]     = 0.0f;
        NLSFr[i][M-1]   = 0.0f; 
        NLSFi[i][M-1]   = 0.0f;
    }
    for(j=0;j<M;j++){
        NLSFr[0][j]     = 0.0f; 
        NLSFi[0][j]     = 0.0f;
        NLSFr[N-1][j]   = 0.0f; 
        NLSFi[N-1][j]   = 0.0f; 	                  
    }
    break;
}/*BC Switch*/  

}                

void mexFunction(int nlhs, mxArray *plhs[],
int nrhs, const mxArray *prhs[])
{
    
/*Input: U,V,s,a,h2,BC,chunk_size,k*/    
    
int    i, j, c, N, M, chunk_size, BC;
float h2, a, s, k, k2,k6, ah2;
double *vUoldr, *vUoldi, *vV;
double **Uoldr, **Uoldi, **V;
double *vUnewr, *vUnewi;
float **fUoldr, **fUoldi, **fV;
float **Utmpr,  **Utmpi;
float **ktmpr,  **ktmpi, **ktotr, **ktoti;

/* Find the dimensions of the vector */
N = mxGetN(prhs[0]);
M = mxGetM(prhs[0]);

/* Create an mxArray for the output data */
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
vV      = mxGetPr(prhs[1]);

Uoldr   = (double **) malloc(sizeof(double*)*N);
Uoldi   = (double **) malloc(sizeof(double*)*N);
V       = (double **) malloc(sizeof(double*)*N);

fUoldr  = (float **) malloc(sizeof(float*)*N);
fUoldi  = (float **) malloc(sizeof(float*)*N);
fV      = (float **) malloc(sizeof(float*)*N);

Utmpr   = (float **) malloc(sizeof(float*)*N);
Utmpi   = (float **) malloc(sizeof(float*)*N);
ktmpr   = (float **) malloc(sizeof(float*)*N);
ktmpi   = (float **) malloc(sizeof(float*)*N);
ktotr   = (float **) malloc(sizeof(float*)*N);
ktoti   = (float **) malloc(sizeof(float*)*N);


for (i=0;i<N;i++){   
    V[i]       = vV+M*i;            
    Uoldr[i]   = vUoldr+M*i;   
    Uoldi[i]   = vUoldi+M*i; 
    fUoldr[i]  = (float *) malloc(sizeof(float)*M);
    fUoldi[i]  = (float *) malloc(sizeof(float)*M);
    fV[i]      = (float *) malloc(sizeof(float)*M);
    Utmpr[i]   = (float *) malloc(sizeof(float)*M);
    Utmpi[i]   = (float *) malloc(sizeof(float)*M);
    ktmpr[i]   = (float *) malloc(sizeof(float)*M);
    ktmpi[i]   = (float *) malloc(sizeof(float)*M);
    ktotr[i]   = (float *) malloc(sizeof(float)*M);
    ktoti[i]   = (float *) malloc(sizeof(float)*M);  
}

/*Convert to floats:*/
for(i=0;i<N;i++){
  for(j=0;j<M;j++){
   fUoldr[i][j] = (float)Uoldr[i][j];
   fUoldi[i][j] = (float)Uoldi[i][j];
       fV[i][j] = (float)V[i][j];
  }
}

s          = (float)mxGetScalar(prhs[2]);
a          = (float)mxGetScalar(prhs[3]);
h2         = (float)mxGetScalar(prhs[4]);
BC         = (int)mxGetScalar(prhs[5]);
chunk_size = (int)mxGetScalar(prhs[6]);
k          = (float)mxGetScalar(prhs[7]);

ah2 = a/h2;
k2 = k/2.0f;
k6 = k/6.0f;

for (c = 0; c<chunk_size; c++)
{   
    NLSE2D_STEP(ktotr,ktoti,fUoldr,fUoldi,fV,s,ah2,BC,N,M);
    add_matrix(Utmpr,Utmpi,fUoldr,fUoldi,ktotr,ktoti,k2,N,M);          
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,fV,s,ah2,BC,N,M);
    add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N,M);   
    add_matrix(Utmpr,Utmpi,fUoldr,fUoldi,ktmpr,ktmpi,k2,N,M);   
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,fV,s,ah2,BC,N,M);  
    add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N,M);   
    add_matrix(Utmpr,Utmpi,fUoldr,fUoldi,ktmpr,ktmpi,k,N,M);  
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,fV,s,ah2,BC,N,M);       
    add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0f,N,M);   
    add_matrix(fUoldr,fUoldi,fUoldr,fUoldi,ktotr,ktoti,k6,N,M);  

}/*End of c chunk_size*/

vUnewr = mxGetPr(plhs[0]);
vUnewi = mxGetPi(plhs[0]);


/*Copy 2D array into result vector*/
for(i=0;i<N;i++){
    for(j=0;j<M;j++){
        vUnewr[M*i+j] = (double)fUoldr[i][j];
        vUnewi[M*i+j] = (double)fUoldi[i][j];
    }
}

/*Free up memory:*/
for (i=0;i<N;i++){   
    free(Utmpr[i]);
    free(Utmpi[i]);
    free(ktmpr[i]);
    free(ktmpi[i]);
    free(ktotr[i]);
    free(ktoti[i]);
    
    free(fUoldr[i]);
    free(fUoldi[i]);
    free(fV[i]);
}
free(Uoldr);  free(Uoldi);
free(fUoldr); free(fUoldi);
free(Utmpr);  free(Utmpi);
free(ktmpr);  free(ktmpi);
free(ktotr);  free(ktoti); 
free(V);      free(fV);

if(!mxIsComplex(prhs[0])){
	free(vUoldi);
}

}

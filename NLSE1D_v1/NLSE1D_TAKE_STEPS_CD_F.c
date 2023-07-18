/*----------------------------
NLSE1D_TAKE_STEPS_CD_F.c
Program to integrate a chunk of time steps of the 1D Nonlinear Shrodinger Equation
i*Ut + a*Uxx - V(r)*U + s*|U|^2*U = 0
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

void vec_add(float* Ar,    float* Ai,
             float* Br,    float* Bi,
             float* Cr,    float* Ci,
             float   k,    int N)
{
	int i;	
	for (i = 0; i < N; i++){
	  	Ar[i]  = Br[i] + k*Cr[i];
    	Ai[i]  = Bi[i] + k*Ci[i]; 
	}
}

void compute_F(float* NLSFr, float* NLSFi, 
               float* Utmpr, float* Utmpi, 
               float* V, float s, float ah2, 
               int BC, int N){

int i;	 
float OM;

for (i = 1; i < N-1; i++)
{   
  NLSFr[i] = -ah2*(Utmpi[i+1] - 2*Utmpi[i] + Utmpi[i-1]) 
    - (s*(Utmpr[i]*Utmpr[i] + Utmpi[i]*Utmpi[i]) - V[i])*Utmpi[i];
  NLSFi[i] = ah2*(Utmpr[i+1] - 2*Utmpr[i] + Utmpr[i-1]) 
    + (s*(Utmpr[i]*Utmpr[i] + Utmpi[i]*Utmpi[i]) - V[i])*Utmpr[i];
}
/*Boundary conditions:*/
switch (BC){
    case 1:
      NLSFr[0]   = 0.0f; 
	  NLSFi[0]   = 0.0f; 
	  NLSFr[N-1] = 0.0f; 
	  NLSFi[N-1] = 0.0f;
      break;
    case 2:      
      OM = (NLSFi[1]*Utmpr[1] - NLSFr[1]*Utmpi[1])/
           (Utmpr[1]*Utmpr[1] + Utmpi[1]*Utmpi[1]);
                                        
      NLSFr[0]  = -OM*Utmpi[0];
      NLSFi[0]  =  OM*Utmpr[0];  
      
      OM = (NLSFi[N-2]*Utmpr[N-2] - NLSFr[N-2]*Utmpi[N-2])/
           (Utmpr[N-2]*Utmpr[N-2] + Utmpi[N-2]*Utmpi[N-2]);
                                        
      NLSFr[N-1]  = -OM*Utmpi[N-1];
      NLSFi[N-1]  =  OM*Utmpr[N-1];     
      
      break;
    case 3:
      NLSFr[0]   = - (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpi[0];
	  NLSFi[0]   =   (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpr[0];
	  NLSFr[N-1] = - (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpi[N-1];
	  NLSFi[N-1] =   (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpr[N-1];
      break;
    case 4:
      NLSFr[0] = -ah2*(-Utmpi[3] + 4*Utmpi[2] - 5*Utmpi[1] + 2*Utmpi[0])
                 - (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpi[0];
      NLSFi[0] = ah2*(-Utmpr[3] + 4*Utmpr[2] - 5*Utmpr[1] + 2*Utmpr[0]) 
                 + (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpr[0];
    
      NLSFr[N-1] = -ah2*(-Utmpi[N-4] + 4*Utmpi[N-3] - 5*Utmpi[N-2] + 2*Utmpi[N-1]) 
                   - (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpi[N-1];
      NLSFi[N-1] = ah2*(-Utmpr[N-4] + 4*Utmpr[N-3] - 5*Utmpr[N-2] + 2*Utmpr[N-1]) 
                   + (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpr[N-1];  
      break;
    default:
      NLSFr[0]   = 0.0f; 
	  NLSFi[0]   = 0.0f; 
	  NLSFr[N-1] = 0.0f; 
	  NLSFi[N-1] = 0.0f;
      break;
   }/*BC Switch*/  
	                  
}	                  

void mexFunction(int nlhs, mxArray *plhs[],
int nrhs, const mxArray *prhs[])
{
    
/*Input: U,V,s,a,h2,BC,chunk_size,k*/    
    
int    i, j, N, chunk_size, BC;
float h2, a, s, ah2,k,k2,k6;
double *Uoldr, *Uoldi, *Unewr, *Unewi, *V;
float *fUoldr,*fUoldi,*Utmpr,*Utmpi,*fV;
float *ktmpr, *ktmpi, *ktotr, *ktoti;

/* Find the dimensions of the vector */
N = mxGetN(prhs[0]);
if(N==1){
    N = mxGetM(prhs[0]);
}

/* Create an mxArray for the output data */
plhs[0] = mxCreateDoubleMatrix(N,1,mxCOMPLEX);

/* Retrieve the input data */
Uoldr = mxGetPr(prhs[0]);

if(mxIsComplex(prhs[0])){
	Uoldi = mxGetPi(prhs[0]);
}
else{
	Uoldi = (double*)malloc(N*sizeof(double));
    for(j=0;j<N;j++){
        Uoldi[j] = 0.0f;
    }
}


V     = mxGetPr(prhs[1]);
s     = (float)mxGetScalar(prhs[2]);
a     = (float)mxGetScalar(prhs[3]);
h2    = (float)mxGetScalar(prhs[4]);
BC    =      (int)mxGetScalar(prhs[5]);
chunk_size = (int)mxGetScalar(prhs[6]);
k     = (float)mxGetScalar(prhs[7]);

ah2 = a/h2;
k2  = k/2.0f;
k6  = k/6.0f;

fUoldr  = (float *) malloc(sizeof(float)*N);
fUoldi  = (float *) malloc(sizeof(float)*N);
fV      = (float *) malloc(sizeof(float)*N);
Utmpr   = (float *) malloc(sizeof(float)*N);
Utmpi   = (float *) malloc(sizeof(float)*N);
ktmpr   = (float *) malloc(sizeof(float)*N);
ktmpi   = (float *) malloc(sizeof(float)*N);
ktotr   = (float *) malloc(sizeof(float)*N);
ktoti   = (float *) malloc(sizeof(float)*N);

/*Convert to floats:*/
for(i=0;i<N;i++){
   fUoldr[i] = (float)Uoldr[i];
   fUoldi[i] = (float)Uoldi[i];
       fV[i] = (float)V[i];
}

for (j = 0; j<chunk_size; j++)
{   
compute_F(ktotr,ktoti,fUoldr,fUoldi,fV,s,ah2,BC,N);
  vec_add(Utmpr,Utmpi,fUoldr,fUoldi,ktotr,ktoti,k2,N);            
compute_F(ktmpr,ktmpi,Utmpr,Utmpi,fV,s,ah2,BC,N);
  vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N);
  vec_add(Utmpr,Utmpi,fUoldr,fUoldi,ktmpr,ktmpi,k2,N); 
compute_F(ktmpr,ktmpi,Utmpr,Utmpi,fV,s,ah2,BC,N);    
  vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N);
  vec_add(Utmpr,Utmpi,fUoldr,fUoldi,ktmpr,ktmpi,k,N);       
compute_F(ktmpr,ktmpi,Utmpr,Utmpi,fV,s,ah2,BC,N);         
  vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0f,N);
  vec_add(fUoldr,fUoldi,fUoldr,fUoldi,ktotr,ktoti,k6,N);   
}/*End of j chunk_size*/

Unewr = mxGetPr(plhs[0]);
Unewi = mxGetPi(plhs[0]);

/*Convert to double:*/
for(i=0;i<N;i++){
   Unewr[i] = (double)fUoldr[i];
   Unewi[i] = (double)fUoldi[i];
}

/*Free up memory:*/
free(fUoldr); free(fUoldi);
free(Utmpr); free(Utmpi);
free(ktmpr);  free(ktmpi);
free(ktotr);  free(ktoti); 
free(fV);

if(!mxIsComplex(prhs[0])){
	free(Uoldi);
}


}

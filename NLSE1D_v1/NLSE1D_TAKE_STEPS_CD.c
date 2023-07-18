/*----------------------------
NLSE1D_TAKE_STEPS_CD.c
Program to integrate a chunk of time steps of the 1D Nonlinear Shrodinger Equation
i*Ut + a*Uxx - V(r)*U + s*|U|^2*U = 0
using RK4 + CD in double precision.

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

void vec_add(double* Ar,    double* Ai,
             double* Br,    double* Bi,
             double* Cr,    double* Ci,
             double   k,    int N)
{
	int i = 0;	
	for (i = 0; i < N; i++){
	  	Ar[i]  = Br[i] + k*Cr[i];
    	Ai[i]  = Bi[i] + k*Ci[i]; 
	}
}

void compute_F(double* NLSFr, double* NLSFi, 
               double* Utmpr, double* Utmpi, 
               double* V, double s, double ah2, 
               int BC, int N){

int i;	
double OM;

for (i = 1; i < N-1; i++)
{   
  NLSFr[i] = -ah2*(Utmpi[i+1] -2*Utmpi[i] + Utmpi[i-1]) 
    - (s*(Utmpr[i]*Utmpr[i] + Utmpi[i]*Utmpi[i]) - V[i])*Utmpi[i];
  NLSFi[i] = ah2*(Utmpr[i+1] -2*Utmpr[i]  + Utmpr[i-1]) 
    + (s*(Utmpr[i]*Utmpr[i] + Utmpi[i]*Utmpi[i]) - V[i])*Utmpr[i];
}
/*Boundary conditions:*/
switch (BC){
    case 1: /*Dirichlet*/
      NLSFr[0]   = 0.0; 
	  NLSFi[0]   = 0.0; 
	  NLSFr[N-1] = 0.0; 
	  NLSFi[N-1] = 0.0;
      break;
    case 2:  /*MSD*/            
      OM = (NLSFi[1]*Utmpr[1] - NLSFr[1]*Utmpi[1])/
           (Utmpr[1]*Utmpr[1] + Utmpi[1]*Utmpi[1]);
                                        
      NLSFr[0]  = -OM*Utmpi[0];
      NLSFi[0]  =  OM*Utmpr[0];  
      
      OM = (NLSFi[N-2]*Utmpr[N-2] - NLSFr[N-2]*Utmpi[N-2])/
           (Utmpr[N-2]*Utmpr[N-2] + Utmpi[N-2]*Utmpi[N-2]);
                                        
      NLSFr[N-1]  = -OM*Utmpi[N-1];
      NLSFi[N-1]  =  OM*Utmpr[N-1];         
      break;
    case 3: /*Uxx=0*/
      NLSFr[0]   = - (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpi[0];
	  NLSFi[0]   =   (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpr[0];
	  NLSFr[N-1] = - (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpi[N-1];
	  NLSFi[N-1] =   (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpr[N-1];
      break;
    case 4: /*One-Sided diff*/
      NLSFr[0] = -ah2*(-Utmpi[3] + 4*Utmpi[2] - 5*Utmpi[1] + 2*Utmpi[0])
                 - (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpi[0];
      NLSFi[0] = ah2*(-Utmpr[3] + 4*Utmpr[2] - 5*Utmpr[1]+ 2*Utmpr[0]) 
                 + (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpr[0];
    
      NLSFr[N-1] = -ah2*(-Utmpi[N-4] + 4*Utmpi[N-3] - 5*Utmpi[N-2] + 2*Utmpi[N-1]) 
                   - (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpi[N-1];
      NLSFi[N-1] = ah2*(-Utmpr[N-4] + 4*Utmpr[N-3] - 5*Utmpr[N-2] + 2*Utmpr[N-1]) 
                   + (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpr[N-1];  
      break;      
    default:
      NLSFr[0]   = 0.0; 
	  NLSFi[0]   = 0.0; 
	  NLSFr[N-1] = 0.0; 
	  NLSFi[N-1] = 0.0;
      break;
   }/*BC Switch*/              
}

void mexFunction(int nlhs, mxArray *plhs[],
int nrhs, const mxArray *prhs[])
{
    
/*Input: U,V,s,a,h2,BC,chunk_size,k*/    
    
int    i, j, N, chunk_size, BC;
double h2, a, s, ah2, k,k2,k6;
double *Uoldr, *Uoldi, *Unewr, *Unewi;
double *Utmpr,*Utmpi,*V;
double *ktmpr, *ktmpi, *ktotr, *ktoti;

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
        Uoldi[j] = 0.0;
    }
}

V     =               mxGetPr(prhs[1]);
s     =   (double)mxGetScalar(prhs[2]);
a     =   (double)mxGetScalar(prhs[3]);
h2    =   (double)mxGetScalar(prhs[4]);
BC    =      (int)mxGetScalar(prhs[5]);
chunk_size = (int)mxGetScalar(prhs[6]);
k     =   (double)mxGetScalar(prhs[7]);

Utmpr   = (double *) malloc(sizeof(double)*N);
Utmpi   = (double *) malloc(sizeof(double)*N);
ktmpr   = (double *) malloc(sizeof(double)*N);
ktmpi   = (double *) malloc(sizeof(double)*N);
ktotr   = (double *) malloc(sizeof(double)*N);
ktoti   = (double *) malloc(sizeof(double)*N);

/*Save divisions*/
ah2 = a/h2;
k2  = k/2.0;
k6  = k/6.0;

for (j = 0; j<chunk_size; j++)
{   
compute_F(ktotr,ktoti,Uoldr,Uoldi,V,s,ah2,BC,N);
  vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N);            
compute_F(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);
  vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N);
  vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N); 
compute_F(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);    
  vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N);
  vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N);       
compute_F(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);         
  vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0,N);
  vec_add(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N);   
}/*End of j chunk_size*/

Unewr = mxGetPr(plhs[0]);
Unewi = mxGetPi(plhs[0]);

for(i=0;i<N;i++){
   Unewr[i] = Uoldr[i];
   Unewi[i] = Uoldi[i];
}

/*Free up memory:*/
free(Utmpr);  
free(Utmpi);
free(ktmpr);  
free(ktmpi);
free(ktotr);  
free(ktoti); 

if(!mxIsComplex(prhs[0])){
  	free(Uoldi);
}

}

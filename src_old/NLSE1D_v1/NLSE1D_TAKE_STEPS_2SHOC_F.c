/*----------------------------
NLSE1D_TAKE_STEPS_2SHOC_F.c
Program to integrate a chunk of time steps of the 1D Nonlinear Shrodinger Equation
i*Ut + a*Uxx - V(r)*U + s*|U|^2*U = 0
using RK4 + 2SHOC in single precision.

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
	int i = 0;	
	for (i = 0; i < N; i++){
	  	Ar[i]  = Br[i] + k*Cr[i];
    	Ai[i]  = Bi[i] + k*Ci[i]; 
	}
}

void compute_F(float* NLSFr, float* NLSFi, 
               float* Utmpr, float* Utmpi, 
			   float* dx2r,  float* dx2i,
               float* V,     float  s, 
               float  a,     float  l_a, 
               float  a76,   float a112, 
               float  lh2,
               int     BC,    int     N    ){

int i;
float OM,A,Nb,Nb1;

/*Compute second-order Laplacian*/
for (i = 1; i < N-1; i++)
{ 
	dx2r[i] =  (Utmpr[i+1] - 2*Utmpr[i] + Utmpr[i-1])*lh2; 
	dx2i[i] =  (Utmpi[i+1] - 2*Utmpi[i] + Utmpi[i-1])*lh2;
}
/*Boundary Conditions*/
switch (BC){
    case 1:
       dx2r[0]   = -l_a*(s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpr[0];
	   dx2i[0]   = -l_a*(s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpi[0]; 
	   dx2r[N-1] = -l_a*(s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpr[N-1]; 
	   dx2i[N-1] = -l_a*(s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpi[N-1]; 
       break;
    case 2:
       A =  (dx2r[1]*Utmpr[1]  + dx2i[1]*Utmpi[1])/
            (Utmpr[1]*Utmpr[1] + Utmpi[1]*Utmpi[1]);       
       Nb  = s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0];
       Nb1 = s*(Utmpr[1]*Utmpr[1] + Utmpi[1]*Utmpi[1]) - V[1];	   
		
	   dx2r[0]  = (A + l_a*(Nb1-Nb))*Utmpr[0];
       dx2i[0]  = (A + l_a*(Nb1-Nb))*Utmpi[0];    
      
	   A =  (dx2r[N-2]*Utmpr[N-2]  + dx2i[N-2]*Utmpi[N-2])/
            (Utmpr[N-2]*Utmpr[N-2] + Utmpi[N-2]*Utmpi[N-2]);       
       Nb  = s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1];
       Nb1 = s*(Utmpr[N-2]*Utmpr[N-2] + Utmpi[N-2]*Utmpi[N-2]) - V[N-2];	   
		
	   dx2r[N-1]  = (A + l_a*(Nb1-Nb))*Utmpr[N-1];
       dx2i[N-1]  = (A + l_a*(Nb1-Nb))*Utmpi[N-1];  
       break;
    case 3: 
       dx2r[0]   = 0.0f; 
	   dx2i[0]   = 0.0f; 
	   dx2r[N-1] = 0.0f; 
	   dx2i[N-1] = 0.0f;
       break;
    case 4:
       dx2r[0]   = (-Utmpr[3] + 4*Utmpr[2] - 5*Utmpr[1] + 2*Utmpr[0])*lh2;
       dx2i[0]   = (-Utmpi[3] + 4*Utmpi[2] - 5*Utmpi[1] + 2*Utmpi[0])*lh2;    
       dx2r[N-1] = (-Utmpr[N-4] + 4*Utmpr[N-3] - 5*Utmpr[N-2] + 2*Utmpr[N-1])*lh2;                     
       dx2i[N-1] = (-Utmpi[N-4] + 4*Utmpi[N-3] - 5*Utmpi[N-2] + 2*Utmpi[N-1])*lh2;    
       break;
    default:
 	   dx2r[0]   = 0.0f; 
	   dx2i[0]   = 0.0f; 
	   dx2r[N-1] = 0.0f; 
	   dx2i[N-1] = 0.0f;
       break;
}/*BC Switch*/


/* Now compute Ut: */
for (i = 1; i < N-1; i++)
{   
  NLSFr[i] =  a112*(dx2i[i+1] + dx2i[i-1]) - a76*dx2i[i]
             - (s*(Utmpr[i]*Utmpr[i] + Utmpi[i]*Utmpi[i]) - V[i])*Utmpi[i];
  NLSFi[i] =  a76*dx2r[i] - a112*(dx2r[i+1] + dx2r[i-1])
             + (s*(Utmpr[i]*Utmpr[i] + Utmpi[i]*Utmpi[i]) - V[i])*Utmpr[i];
}
/*Boundary Conditions*/
   switch (BC){
    case 1:
      NLSFr[0]   = 0.0f; 
	  NLSFi[0]   = 0.0f; 
	  NLSFr[N-1] = 0.0f; 
	  NLSFi[N-1] = 0.0f;
      break;
    case 2:   /*MSD*/   
      OM =  (NLSFi[1]*Utmpr[1] -  NLSFr[1]*Utmpi[1])/
            (Utmpr[1]*Utmpr[1] + Utmpi[1]*Utmpi[1]);
                                        
      NLSFr[0]  = -OM*Utmpi[0];
      NLSFi[0]  =  OM*Utmpr[0];

      OM =  (NLSFi[N-2]*Utmpr[N-2] -  NLSFr[N-2]*Utmpi[N-2])/
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
      NLSFr[0] = -a*(-Utmpi[3] + 4*Utmpi[2] - 5*Utmpi[1] + 2*Utmpi[0])*lh2
                 - (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpi[0];
      NLSFi[0] = a*(-Utmpr[3] + 4*Utmpr[2] - 5*Utmpr[1] + 2*Utmpr[0])*lh2 
                 + (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpr[0];
    
      NLSFr[N-1] = -a*(-Utmpi[N-4] + 4*Utmpi[N-3] - 5*Utmpi[N-2] + 2*Utmpi[N-1])*lh2 
                   - (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpi[N-1];
      NLSFi[N-1] = a*(-Utmpr[N-4] + 4*Utmpr[N-3] - 5*Utmpr[N-2] + 2*Utmpr[N-1])*lh2 
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
float h2, a, l_a, a76,a112,s, k,k2,k6,lh2;
double *Uoldr, *Uoldi, *Unewr, *Unewi, *V;
float *fUoldr,*fUoldi,*fUtmpr,*fUtmpi,*fV;
float *ktmpr, *ktmpi, *ktotr, *ktoti,*fdx2r,*fdx2i;

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


V     = mxGetPr(prhs[1]);
s    = (float)mxGetScalar(prhs[2]);
a     = (float)mxGetScalar(prhs[3]);
h2    = (float)mxGetScalar(prhs[4]);
BC    =      (int)mxGetScalar(prhs[5]);
chunk_size = (int)mxGetScalar(prhs[6]);
k     = (float)mxGetScalar(prhs[7]);

fUoldr  = (float *) malloc(sizeof(float)*N);
fUoldi  = (float *) malloc(sizeof(float)*N);
fV      = (float *) malloc(sizeof(float)*N);

fUtmpr  = (float *) malloc(sizeof(float)*N);
fUtmpi  = (float *) malloc(sizeof(float)*N);
fdx2r   = (float *) malloc(sizeof(float)*N);
fdx2i   = (float *) malloc(sizeof(float)*N);
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

l_a  = 1.0f/a;
lh2  = 1.0f/h2;
k2   = k/2.0f;
k6   = k/6.0f;
a76  = a*(7.0f/6.0f);
a112 = a*(1.0f/12.0f);


for (j = 0; j<chunk_size; j++)
{   	
compute_F(ktotr, ktoti,fUoldr,fUoldi,fdx2r,fdx2i,fV,s,a,l_a,a76,a112,lh2,BC,N);
/*Evaluate Utmp:*/
 vec_add(fUtmpr,fUtmpi,fUoldr,fUoldi,ktotr,ktoti,k2,N);          
compute_F(ktmpr, ktmpi,fUtmpr,fUtmpi,fdx2r,fdx2i,fV,s,a,l_a,a76,a112,lh2,BC,N);
/*Collect k and evaluate new Utmp*/
 vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N);
 vec_add(fUtmpr,fUtmpi,fUoldr,fUoldi,ktmpr,ktmpi,k2,N); 
compute_F(ktmpr, ktmpi,fUtmpr,fUtmpi,fdx2r,fdx2i,fV,s,a,l_a,a76,a112,lh2,BC,N);    
/*Collect k and evaluate new Utmp again*/
 vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N);
 vec_add(fUtmpr,fUtmpi,fUoldr,fUoldi,ktmpr,ktmpi,k,N);       
compute_F(ktmpr, ktmpi,fUtmpr,fUtmpi,fdx2r,fdx2i,fV,s,a,l_a,a76,a112,lh2,BC,N);          
/*Collect k and evaluate new step*/
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
free(fUtmpr); free(fUtmpi);
free(fdx2r);  free(fdx2i);
free(ktmpr);  free(ktmpi);
free(ktotr);  free(ktoti); 
free(fV);

if(!mxIsComplex(prhs[0])){
	free(Uoldi);
}


}

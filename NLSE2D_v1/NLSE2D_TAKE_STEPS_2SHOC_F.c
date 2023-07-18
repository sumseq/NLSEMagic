/*----------------------------
NLSE2D_TAKE_STEPS_2SHOC_F.c
Program to integrate a chunk of time steps of the 2D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy) - V(r)*U + s*|U|^2*U = 0
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

void add_matrix(float** Ar,    float** Ai,
                float** Br,    float** Bi,
                float** Cr,    float** Ci,
                float    k,    int N,  int M)
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
                 float** V,     
				 float** Dr,    float** Di,
				 float s,       float a, 
                 float a1_6h2,  float a1_12, 
                 float lh2,     float l_a,
                 int BC, int N, int M){

int i,j,k,i1,j1;	
float A,Nb1,Nb,OM;
int msd_x0[4] = {0,0,  N-1,N-1};
int msd_y0[4] = {0,M-1,0,  M-1};
int msd_x1[4] = {1,1,  N-2,N-2};
int msd_y1[4] = {1,M-2,1,  M-2};

/*First compute laplacian*/
for (i = 1; i < N-1; i++)
{       
  for(j = 1;j < M-1;j++)
  {
    Di[i][j] = (Utmpi[i+1][j] + Utmpi[i-1][j] - 4*Utmpi[i][j] + 
                Utmpi[i][j+1] + Utmpi[i][j-1])*lh2;   
    Dr[i][j] = (Utmpr[i+1][j] + Utmpr[i-1][j] - 4*Utmpr[i][j] + 
                Utmpr[i][j+1] + Utmpr[i][j-1])*lh2; 
  }    
}
/*Boundary Conditions*/
switch (BC){
   case 1: /*Dirichlet*/
    for(i=0;i<N;i++){
		Dr[i][0]     = l_a*(V[i][0]   - s*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]))*Utmpr[i][0];
		Di[i][0]     = l_a*(V[i][0]   - s*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]))*Utmpi[i][0];
		Dr[i][M-1]   = l_a*(V[i][M-1] - s*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1]))*Utmpr[i][M-1]; 
		Di[i][M-1]   = l_a*(V[i][M-1] - s*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1]))*Utmpi[i][M-1];
    }
	for(j=0;j<M;j++){
		Dr[0][j]     = l_a*(V[0][j]   - s*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]))*Utmpr[0][j]; 
		Di[0][j]     = l_a*(V[0][j]   - s*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]))*Utmpi[0][j];
		Dr[N-1][j]   = l_a*(V[N-1][j] - s*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j]))*Utmpr[N-1][j]; 
		Di[N-1][j]   = l_a*(V[N-1][j] - s*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j]))*Utmpi[N-1][j]; 	
    }
    break;
    case 2: /*MSD*/
    /*i loop*/
    for(i=1;i<N-1;i++){      
       A   = (Dr[i][1]*Utmpr[i][1] + Di[i][1]*Utmpi[i][1])/
             (Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1]);       
       Nb  = s*(Utmpr[i][0]*Utmpr[i][0] + Utmpi[i][0]*Utmpi[i][0]) - V[i][0];
       Nb1 = s*(Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1]) - V[i][1];	   
		
	   Dr[i][0]  = (A + l_a*(Nb1-Nb))*Utmpr[i][0];
       Di[i][0]  = (A + l_a*(Nb1-Nb))*Utmpi[i][0];      

       A   = (Dr[i][M-2]*Utmpr[i][M-2] + Di[i][M-2]*Utmpi[i][M-2])/
             (Utmpr[i][M-2]*Utmpr[i][M-2] + Utmpi[i][M-2]*Utmpi[i][M-2]);       
       Nb  = s*(Utmpr[i][M-1]*Utmpr[i][M-1] + Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1];
       Nb1 = s*(Utmpr[i][M-2]*Utmpr[i][M-2] + Utmpi[i][M-2]*Utmpi[i][M-2]) - V[i][M-2];	   
		
	   Dr[i][M-1]  = (A + l_a*(Nb1-Nb))*Utmpr[i][M-1];
       Di[i][M-1]  = (A + l_a*(Nb1-Nb))*Utmpi[i][M-1]; 
    }
    /*j loop*/
    for(j=1;j<M-1;j++){
        A   = (Dr[1][j]*Utmpr[1][j] + Di[1][j]*Utmpi[1][j])/
             (Utmpr[1][j]*Utmpr[1][j] + Utmpi[1][j]*Utmpi[1][j]);       
        Nb  = s*(Utmpr[0][j]*Utmpr[0][j] + Utmpi[0][j]*Utmpi[0][j]) - V[0][j];
        Nb1 = s*(Utmpr[1][j]*Utmpr[1][j] + Utmpi[1][j]*Utmpi[1][j]) - V[1][j];	   
		
	    Dr[0][j]  = (A + l_a*(Nb1-Nb))*Utmpr[0][j];
        Di[0][j]  = (A + l_a*(Nb1-Nb))*Utmpi[0][j];   

        A   = (Dr[N-2][j]*Utmpr[N-2][j] + Di[N-2][j]*Utmpi[N-2][j])/
             (Utmpr[N-2][j]*Utmpr[N-2][j] + Utmpi[N-2][j]*Utmpi[N-2][j]);       
        Nb  = s*(Utmpr[N-1][j]*Utmpr[N-1][j] + Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j];
        Nb1 = s*(Utmpr[N-2][j]*Utmpr[N-2][j] + Utmpi[N-2][j]*Utmpi[N-2][j]) - V[N-2][j];	   
		
	    Dr[N-1][j]  = (A + l_a*(Nb1-Nb))*Utmpr[N-1][j];
        Di[N-1][j]  = (A + l_a*(Nb1-Nb))*Utmpi[N-1][j];       
    }
    /*Now do 4 corners*/
    for(k=0;k<4;k++){          
        i  = msd_x0[k];
        j  = msd_y0[k];
        i1 = msd_x1[k];
        j1 = msd_y1[k];        
    
        A   = (Dr[i1][j1]*Utmpr[i1][j1] + Di[i1][j1]*Utmpi[i1][j1])/
              (Utmpr[i1][j1]*Utmpr[i1][j1] + Utmpi[i1][j1]*Utmpi[i1][j1]);       
        Nb  = s*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]) - V[i][j];
        Nb1 = s*(Utmpr[i1][j1]*Utmpr[i1][j1] + Utmpi[i1][j1]*Utmpi[i1][j1]) - V[i1][j1];	   
		 
	    Dr[i][j]  = (A + l_a*(Nb1-Nb))*Utmpr[i][j];
        Di[i][j]  = (A + l_a*(Nb1-Nb))*Utmpi[i][j];    
    }    
    break;
    case 3:
    for(i=0;i<N;i++){
		Dr[i][0]     = 0.0f; 
		Di[i][0]     = 0.0f;
		Dr[i][M-1]   = 0.0f; 
		Di[i][M-1]   = 0.0f;
    }
	for(j=0;j<M;j++){
		Dr[0][j]     = 0.0f; 
		Di[0][j]     = 0.0f;
		Dr[N-1][j]   = 0.0f; 
		Di[N-1][j]   = 0.0f; 	
    }
    break;
    default:
    for(i=0;i<N;i++){
		Dr[i][0]     = 0.0f; 
		Di[i][0]     = 0.0f;
		Dr[i][M-1]   = 0.0f; 
		Di[i][M-1]   = 0.0f;
    }
	for(j=0;j<M;j++){
		Dr[0][j]     = 0.0f; 
		Di[0][j]     = 0.0f;
		Dr[N-1][j]   = 0.0f; 
		Di[N-1][j]   = 0.0f; 	
    }
    break;
}

/*Now compute 2SHOC*/
for (i=1;i<N-1;i++)
{       
  for(j=1;j<M-1;j++)
  {

NLSFr[i][j] = -a*Di[i][j] + a1_12*(Di[i+1][j] + Di[i-1][j] + 
                                   Di[i][j+1] + Di[i][j-1]) 
               - a1_6h2*(Utmpi[i+1][j+1] + Utmpi[i+1][j-1] - 
         4*Utmpi[i][j] + Utmpi[i-1][j+1] + Utmpi[i-1][j-1])
               - (s*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]) 
                      - V[i][j])*Utmpi[i][j];
    
NLSFi[i][j] =  a*Dr[i][j] - a1_12*(Dr[i+1][j] + Dr[i-1][j] + 
                                   Dr[i][j+1] + Dr[i][j-1])
               + a1_6h2*(Utmpr[i+1][j+1] + Utmpr[i+1][j-1] - 
         4*Utmpr[i][j] + Utmpr[i-1][j+1] + Utmpr[i-1][j-1]) 
               + (s*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]) 
                      - V[i][j])*Utmpr[i][j];
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
       NLSFr[i][0]   = - (s*(Utmpr[i][0]*Utmpr[i][0] + 
                           Utmpi[i][0]*Utmpi[i][0]) - V[i][0])*Utmpi[i][0];
	   NLSFi[i][0]   = (s*(Utmpr[i][0]*Utmpr[i][0] + 
                           Utmpi[i][0]*Utmpi[i][0]) - V[i][0])*Utmpr[i][0];
       NLSFr[i][M-1] = - (s*(Utmpr[i][M-1]*Utmpr[i][M-1] + 
                   Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1])*Utmpi[i][M-1];
	   NLSFi[i][M-1] =   (s*(Utmpr[i][M-1]*Utmpr[i][M-1] + 
                   Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1])*Utmpr[i][M-1];
    }    
    for(j=0;j<M;j++){
       NLSFr[0][j]   = - (s*(Utmpr[0][j]*Utmpr[0][j] + 
                          Utmpi[0][j]*Utmpi[0][j]) - V[0][j])*Utmpi[0][j];
	   NLSFi[0][j]   =   (s*(Utmpr[0][j]*Utmpr[0][j] + 
                          Utmpi[0][j]*Utmpi[0][j]) - V[0][j])*Utmpr[0][j];
       NLSFr[N-1][j] = - (s*(Utmpr[N-1][j]*Utmpr[N-1][j] + 
                   Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j])*Utmpi[N-1][j];
	   NLSFi[N-1][j] =   (s*(Utmpr[N-1][j]*Utmpr[N-1][j] + 
                   Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j])*Utmpr[N-1][j];
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

void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    
/*Input: U,V,s,a,h2,BC,chunk_size,k*/    
    
int    i, j, c, N, M, chunk_size, BC;
float h2,lh2, a, a1_6h2, a1_12, s, k,k2,k6,l_a;
double *vUoldr, *vUoldi, *vV;
double **Uoldr, **Uoldi, **V;
double *vUnewr, *vUnewi;
float **fUoldr, **fUoldi, **fV;
float **Utmpr, **Utmpi, **Dr, **Di;
float **ktmpr, **ktmpi, **ktotr, **ktoti;


/* Find the dimensions of the vector */
N = mxGetN(prhs[0]);
M = mxGetM(prhs[0]);
/*printf("N: %d, M: %d\n",N,M);*/
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
vV     = mxGetPr(prhs[1]);


/*Convert input data into 2D arrays*/
Uoldr   = (double **) malloc(sizeof(double*)*N);
Uoldi   = (double **) malloc(sizeof(double*)*N);
V       = (double **) malloc(sizeof(double*)*N);

fUoldr  = (float **) malloc(sizeof(float*)*N);
fUoldi  = (float **) malloc(sizeof(float*)*N);
fV      = (float **) malloc(sizeof(float*)*N);

Dr      = (float **) malloc(sizeof(float*)*N);
Di      = (float **) malloc(sizeof(float*)*N);
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
    Dr[i]      = (float *) malloc(sizeof(float)*M);
    Di[i]      = (float *) malloc(sizeof(float)*M);
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

lh2    = 1.0f/h2;
k2     = k/2.0f;
k6     = k/6.0f;
a1_6h2 = a*(1.0f/(6.0f*h2));
a1_12  = a*(1.0f/12.0f);
	
for (c = 0; c<chunk_size; c++)
{   

    NLSE2D_STEP(ktotr,ktoti,fUoldr,fUoldi,fV,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);
    add_matrix(Utmpr,Utmpi,fUoldr,fUoldi,ktotr,ktoti,k2,N,M);          
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,fV,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);
    add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N,M);   
    add_matrix(Utmpr,Utmpi,fUoldr,fUoldi,ktmpr,ktmpi,k2,N,M);   
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,fV,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);  
    add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N,M);   
    add_matrix(Utmpr,Utmpi,fUoldr,fUoldi,ktmpr,ktmpi,k,N,M);  
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,fV,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);       
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
    free(Dr[i]);
    free(Di[i]);
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
free(Dr);     free(Di);

if(!mxIsComplex(prhs[0])){
	free(vUoldi);
}

}

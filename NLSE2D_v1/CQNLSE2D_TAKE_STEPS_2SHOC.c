/*----------------------------
Program to integrate a chunk of time steps of the 2D CQ Nonlinear Shrodinger Equation
NOTE!  NEW MSD NOT IMPLEMENTED HERE!
i*Ut + a*(Uxx + Uyy) - V(r)*U + (s1*|U|^2 + s2*|U|^4)*U = 0
using RK4 + 2SHOC in double precision.

Ronald M Caplan
Computational Science Research Center
San Diego State University

INPUT:
(U,V,s,a,h2,BC,chunk_size,k)
U  = Current solution matrix
V  = External Potential matrix
s1 = Nonlinearity paramater 1
s2 = Nonlinearity paramater 2
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

void add_matrix(double** Ar,    double** Ai,
                double** Br,    double** Bi,
                double** Cr,    double** Ci,
                double    k,    int N,  int M)
{
	int i,j;
	for(i=0;i<N;i++){
	   for(j=0;j<M;j++){
    	Ar[i][j]  = Br[i][j] + k*Cr[i][j];  
    	Ai[i][j]  = Bi[i][j] + k*Ci[i][j];  
	   }
    }
}

void NLSE2D_STEP(double** NLSFr, double** NLSFi, 
                 double** Utmpr, double** Utmpi, 
                 double** V,     
				 double** Dr, double** Di,
				 double s1, double s2, double a, 
                 double a1_6h2, double a1_12, 
                 double lh2,    double l_a, 
                 int BC, int N, int M){

int i,j,k,i1,j1;	
double A,B,NL,OM;
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
		Dr[i][0]     = -l_a*(s1*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]) +
                             s2*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0])*
                                (Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]) - V[i][0])*Utmpr[i][0];
		Di[i][0]     = -l_a*(s1*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0])+
                             s2*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0])*
                                (Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]) - V[i][0])*Utmpi[i][0];
		Dr[i][M-1]   = -l_a*(s1*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1])+
                             s2*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1])*
                                (Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1])*Utmpr[i][M-1]; 
		Di[i][M-1]   = -l_a*(s1*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1])+
                             s2*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1])*
                                (Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1])*Utmpi[i][M-1];
    }
	for(j=0;j<M;j++){
		Dr[0][j]     = -l_a*(s1*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]) + 
                             s2*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j])*
                                (Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]) - V[0][j])*Utmpr[0][j]; 
		Di[0][j]     = -l_a*(s1*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]) + 
                             s2*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j])*
                                (Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j])- V[0][j])*Utmpi[0][j];
		Dr[N-1][j]   = -l_a*(s1*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j])+ 
                             s2*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j])*
                                (Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j])*Utmpr[N-1][j]; 
		Di[N-1][j]   = -l_a*(s1*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j])+ 
                             s2*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j])*
                                (Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j])*Utmpi[N-1][j]; 	
    }
    break;
    case 2: /*MSD*/
    /*i loop*/
    for(i=1;i<N-1;i++){        
        A = (Utmpi[i][1]*Utmpr[i][0] - Utmpr[i][1]*Utmpi[i][0])/
            (Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1]);        
        B = (Utmpi[i][1]*Utmpi[i][0] + Utmpr[i][1]*Utmpr[i][0])/
            (Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1]);
        NL = ( (V[i][0]-V[i][1]) 
             + s1*(Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1]) 
             + s2*(Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1])*
                  (Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1])
             - (s1*(Utmpr[i][0]*Utmpr[i][0] + Utmpi[i][0]*Utmpi[i][0]) 
             + s2*(Utmpr[i][0]*Utmpr[i][0] + Utmpi[i][0]*Utmpi[i][0])*
                  (Utmpr[i][0]*Utmpr[i][0] + Utmpi[i][0]*Utmpi[i][0]))
                  )/a;
	  
        Dr[i][0]  = A*Di[i][1] + B*Dr[i][1] + NL*Utmpr[i][0];
        Di[i][0]  = B*Di[i][1] - A*Dr[i][1] + NL*Utmpi[i][0];                
        
        A = (Utmpi[i][M-2]*Utmpr[i][M-1] - Utmpr[i][M-2]*Utmpi[i][M-1])/
            (Utmpr[i][M-2]*Utmpr[i][M-2] + Utmpi[i][M-2]*Utmpi[i][M-2]);        
        B = (Utmpi[i][M-2]*Utmpi[i][M-1] + Utmpr[i][M-2]*Utmpr[i][M-1])/
            (Utmpr[i][M-2]*Utmpr[i][M-2] + Utmpi[i][M-2]*Utmpi[i][M-2]);
        
        NL = ( (V[i][M-1]-V[i][M-2]) 
             + s1*(Utmpr[i][1]*Utmpr[i][1] + Utmpi[i][1]*Utmpi[i][1]) 
             + s2*(Utmpr[i][M-2]*Utmpr[i][M-2] + Utmpi[i][M-2]*Utmpi[i][M-2])*
                  (Utmpr[i][M-2]*Utmpr[i][M-2] + Utmpi[i][M-2]*Utmpi[i][M-2])
             - (s1*(Utmpr[i][M-1]*Utmpr[i][M-1] + Utmpi[i][M-1]*Utmpi[i][M-1]) 
             +  s2*(Utmpr[i][M-1]*Utmpr[i][M-1] + Utmpi[i][M-1]*Utmpi[i][M-1])*
                  (Utmpr[i][M-1]*Utmpr[i][M-1] + Utmpi[i][M-1]*Utmpi[i][M-1]))
                  )/a;
        Dr[i][M-1]  = A*Di[i][M-2] + B*Dr[i][M-2] + NL*Utmpr[i][M-1];
        Di[i][M-1]  = B*Di[i][M-2] - A*Dr[i][M-2] + NL*Utmpi[i][M-1];     
    }
    /*j loop*/
    for(j=1;j<M-1;j++){
        A = (Utmpi[1][j]*Utmpr[0][j] - Utmpr[1][j]*Utmpi[0][j])/
            (Utmpr[1][j]*Utmpr[1][j] + Utmpi[1][j]*Utmpi[1][j]);        
        B = (Utmpi[1][j]*Utmpi[0][j] + Utmpr[1][j]*Utmpr[0][j])/
            (Utmpr[1][j]*Utmpr[1][j] + Utmpi[1][j]*Utmpi[1][j]);
        NL = ( (V[0][j]-V[1][j]) 
            + s1*(Utmpr[1][j]*Utmpr[1][j] + Utmpi[1][j]*Utmpi[1][j]) 
             + s2*(Utmpr[1][j]*Utmpr[1][j] + Utmpi[1][j]*Utmpi[1][j])*
                  (Utmpr[1][j]*Utmpr[1][j] + Utmpi[1][j]*Utmpi[1][j])
             - (s1*(Utmpr[0][j]*Utmpr[0][j] + Utmpi[0][j]*Utmpi[0][j]) 
             + s2*(Utmpr[0][j]*Utmpr[0][j] + Utmpi[0][j]*Utmpi[0][j])*
                  (Utmpr[0][j]*Utmpr[0][j] + Utmpi[0][j]*Utmpi[0][j]))
                  )/a;
        Dr[0][j]  = A*Di[1][j] + B*Dr[1][j] + NL*Utmpr[0][j];
        Di[0][j]  = B*Di[1][j] - A*Dr[1][j] + NL*Utmpi[0][j];                
        
        A = (Utmpi[N-2][j]*Utmpr[N-1][j] - Utmpr[N-2][j]*Utmpi[N-1][j])/
            (Utmpr[N-2][j]*Utmpr[N-2][j] + Utmpi[N-2][j]*Utmpi[N-2][j]);        
        B = (Utmpi[N-2][j]*Utmpi[N-1][j] + Utmpr[N-2][j]*Utmpr[N-1][j])/
            (Utmpr[N-2][j]*Utmpr[N-2][j] + Utmpi[N-2][j]*Utmpi[N-2][j]);
        NL = ( (V[N-1][j]-V[N-2][j]) 
            + s1*(Utmpr[N-2][j]*Utmpr[N-2][j] + Utmpi[N-2][j]*Utmpi[N-2][j]) 
             + s2*(Utmpr[N-2][j]*Utmpr[N-2][j] + Utmpi[N-2][j]*Utmpi[N-2][j])*
                  (Utmpr[N-2][j]*Utmpr[N-2][j] + Utmpi[N-2][j]*Utmpi[N-2][j])
             - (s1*(Utmpr[N-1][j]*Utmpr[N-1][j] + Utmpi[N-1][j]*Utmpi[N-1][j]) 
             + s2*(Utmpr[N-1][j]*Utmpr[N-1][j] + Utmpi[N-1][j]*Utmpi[N-1][j])*
                  (Utmpr[N-1][j]*Utmpr[N-1][j] + Utmpi[N-1][j]*Utmpi[N-1][j]))
                  )/a;
        Dr[N-1][j]  = A*Di[N-2][j] + B*Dr[N-2][j] + NL*Utmpr[N-1][j];
        Di[N-1][j]  = B*Di[N-2][j] - A*Dr[N-2][j] + NL*Utmpi[N-1][j];        
    }
    /*Now do 4 corners*/
    for(k=0;k<4;k++){          
        i  = msd_x0[k];
        j  = msd_y0[k];
        i1 = msd_x1[k];
        j1 = msd_y1[k];        
    
        A = (Utmpi[i1][j1]*Utmpr[i][j]   - Utmpr[i1][j1]*Utmpi[i][j])/
            (Utmpr[i1][j1]*Utmpr[i1][j1] + Utmpi[i1][j1]*Utmpi[i1][j1]);        
        B = (Utmpi[i1][j1]*Utmpi[i][j]   + Utmpr[i1][j1]*Utmpr[i][j])/
            (Utmpr[i1][j1]*Utmpr[i1][j1] + Utmpi[i1][j1]*Utmpi[i1][j1]);    
        
        NL = ( (V[i][j]-V[i1][j1]) 
            + s1*(Utmpr[i1][j1]*Utmpr[i1][j1] + Utmpi[i1][j1]*Utmpi[i1][j1]) 
             + s2*(Utmpr[i1][j1]*Utmpr[i1][j1] + Utmpi[i1][j1]*Utmpi[i1][j1])*
                  (Utmpr[i1][j1]*Utmpr[i1][j1] + Utmpi[i1][j1]*Utmpi[i1][j1])
             - (s1*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]) 
             + s2*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j])*
                  (Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]))
                  )/a;
        Dr[i][j]  = A*Di[i1][j1]  + B*Dr[i1][j1] + NL*Utmpr[i][j];
        Di[i][j]  = B*Di[i1][j1]  - A*Dr[i1][j1] + NL*Utmpi[i][j];       
    }    
    break;
    case 3:
    for(i=0;i<N;i++){
		Dr[i][0]     = 0.0; 
		Di[i][0]     = 0.0;
		Dr[i][M-1]   = 0.0; 
		Di[i][M-1]   = 0.0;
    }
	for(j=0;j<M;j++){
		Dr[0][j]     = 0.0; 
		Di[0][j]     = 0.0;
		Dr[N-1][j]   = 0.0; 
		Di[N-1][j]   = 0.0; 	
    }
    break;
    default:
    for(i=0;i<N;i++){
		Dr[i][0]     = 0.0; 
		Di[i][0]     = 0.0;
		Dr[i][M-1]   = 0.0; 
		Di[i][M-1]   = 0.0;
    }
	for(j=0;j<M;j++){
		Dr[0][j]     = 0.0; 
		Di[0][j]     = 0.0;
		Dr[N-1][j]   = 0.0; 
		Di[N-1][j]   = 0.0; 	
    }
    break;
}

/*Now compute HOC*/
for (i=1;i<N-1;i++)
{       
  for(j=1;j<M-1;j++)
  {

NLSFr[i][j] = -a*Di[i][j] + a1_12*(Di[i+1][j] + Di[i-1][j] + 
                                   Di[i][j+1] + Di[i][j-1]) 
               - a1_6h2*(Utmpi[i+1][j+1] + Utmpi[i+1][j-1] - 
         4*Utmpi[i][j] + Utmpi[i-1][j+1] + Utmpi[i-1][j-1])
               - (s1*(Utmpr[i][j]*Utmpr[i][j]+Utmpi[i][j]*Utmpi[i][j]) +
                             s2*(Utmpr[i][j]*Utmpr[i][j]+Utmpi[i][j]*Utmpi[i][j])*
                                (Utmpr[i][j]*Utmpr[i][j]+Utmpi[i][j]*Utmpi[i][j]) - V[i][j])*Utmpi[i][j];
    
NLSFi[i][j] =  a*Dr[i][j] - a1_12*(Dr[i+1][j] + Dr[i-1][j] + 
                                   Dr[i][j+1] + Dr[i][j-1])
               + a1_6h2*(Utmpr[i+1][j+1] + Utmpr[i+1][j-1] - 
         4*Utmpr[i][j] + Utmpr[i-1][j+1] + Utmpr[i-1][j-1]) 
               + (s1*(Utmpr[i][j]*Utmpr[i][j]+Utmpi[i][j]*Utmpi[i][j]) +
                             s2*(Utmpr[i][j]*Utmpr[i][j]+Utmpi[i][j]*Utmpi[i][j])*
                                (Utmpr[i][j]*Utmpr[i][j]+Utmpi[i][j]*Utmpi[i][j])  - V[i][j])*Utmpr[i][j];
  }    
}
/*Boundary conditions:*/
switch (BC){
    case 1:
    for(i=0;i<N;i++){
        NLSFr[i][0]     = 0.0; 
        NLSFi[i][0]     = 0.0;
        NLSFr[i][M-1]   = 0.0; 
        NLSFi[i][M-1]   = 0.0;
    }
    for(j=0;j<M;j++){
        NLSFr[0][j]     = 0.0; 
        NLSFi[0][j]     = 0.0;
        NLSFr[N-1][j]   = 0.0; 
        NLSFi[N-1][j]   = 0.0; 	                  
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
       NLSFr[i][0]   = - (s1*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]) +
                             s2*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0])*
                                (Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]) - V[i][0])*Utmpi[i][0];
	   NLSFi[i][0]   = (s1*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]) +
                             s2*(Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0])*
                                (Utmpr[i][0]*Utmpr[i][0]+Utmpi[i][0]*Utmpi[i][0]) - V[i][0])*Utmpr[i][0];
       NLSFr[i][M-1] = - (s1*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1])+
                             s2*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1])*
                                (Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1])*Utmpi[i][M-1];
	   NLSFi[i][M-1] =   (s1*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1])+
                             s2*(Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1])*
                                (Utmpr[i][M-1]*Utmpr[i][M-1]+Utmpi[i][M-1]*Utmpi[i][M-1]) - V[i][M-1])*Utmpr[i][M-1];
    }    
    for(j=0;j<M;j++){
       NLSFr[0][j]   = - (s1*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]) + 
                             s2*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j])*
                                (Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]) - V[0][j])*Utmpi[0][j];
	   NLSFi[0][j]   =   (s1*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]) + 
                             s2*(Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j])*
                                (Utmpr[0][j]*Utmpr[0][j]+Utmpi[0][j]*Utmpi[0][j]) - V[0][j])*Utmpr[0][j];
       NLSFr[N-1][j] = - (s1*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j])+ 
                             s2*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j])*
                                (Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j])*Utmpi[N-1][j];
	   NLSFi[N-1][j] =   (s1*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j])+ 
                             s2*(Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j])*
                                (Utmpr[N-1][j]*Utmpr[N-1][j]+Utmpi[N-1][j]*Utmpi[N-1][j]) - V[N-1][j])*Utmpr[N-1][j];
    }
    break;
    default:
    for(i=0;i<N;i++){
        NLSFr[i][0]     = 0.0; 
        NLSFi[i][0]     = 0.0;
        NLSFr[i][M-1]   = 0.0; 
        NLSFi[i][M-1]   = 0.0;
    }
    for(j=0;j<M;j++){
        NLSFr[0][j]     = 0.0; 
        NLSFi[0][j]     = 0.0;
        NLSFr[N-1][j]   = 0.0; 
        NLSFi[N-1][j]   = 0.0; 	                  
    }
    break;
}/*BC Switch*/  

}	                  

void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    
/*Input: U,V,s,a,h2,BC,chunk_size,k*/    
    
int    i, j, c, N, M, chunk_size, BC;
double h2,lh2, a, a1_6h2, a1_12, s1, s2, k,k2,k6,l_a;
double *vUoldr, *vUoldi,  *vV;
double **Uoldr, **Uoldi, *Unewr, *Unewi, **V;
double **Utmpr, **Utmpi, **Dr, **Di;
double **ktmpr, **ktmpi, **ktotr, **ktoti;

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
        vUoldi[i] = 0;
    }
}
vV     = mxGetPr(prhs[1]);


/*Convert input data into 2D arrays*/
Uoldr   = (double **) malloc(sizeof(double*)*N);
Uoldi   = (double **) malloc(sizeof(double*)*N);
V       = (double **) malloc(sizeof(double*)*N);
Dr      = (double **) malloc(sizeof(double*)*N);
Di      = (double **) malloc(sizeof(double*)*N);
Utmpr   = (double **) malloc(sizeof(double*)*N);
Utmpi   = (double **) malloc(sizeof(double*)*N);
ktmpr   = (double **) malloc(sizeof(double*)*N);
ktmpi   = (double **) malloc(sizeof(double*)*N);
ktotr   = (double **) malloc(sizeof(double*)*N);
ktoti   = (double **) malloc(sizeof(double*)*N);


for (i=0;i<N;i++){   
    V[i]       = vV+M*i;            
    Uoldr[i]   = vUoldr+M*i;   
    Uoldi[i]   = vUoldi+M*i;    
    Dr[i]      = (double *) malloc(sizeof(double)*M);
    Di[i]      = (double *) malloc(sizeof(double)*M);
    Utmpr[i]   = (double *) malloc(sizeof(double)*M);
    Utmpi[i]   = (double *) malloc(sizeof(double)*M);
    ktmpr[i]   = (double *) malloc(sizeof(double)*M);
    ktmpi[i]   = (double *) malloc(sizeof(double)*M);
    ktotr[i]   = (double *) malloc(sizeof(double)*M);
    ktoti[i]   = (double *) malloc(sizeof(double)*M);  
}


s1    = mxGetScalar(prhs[2]);
s2    = mxGetScalar(prhs[3]);
a     = mxGetScalar(prhs[4]);
h2    = mxGetScalar(prhs[5]);
BC    = (int)mxGetScalar(prhs[6]);
chunk_size = (int)mxGetScalar(prhs[7]);
k     = mxGetScalar(prhs[8]);

l_a    = 1.0/a;
lh2    = 1.0/h2;
k2     = k/2.0;
k6     = k/6.0;
a1_6h2 = a*(1.0/(6.0*h2));
a1_12  = a*(1.0/12.0);
	
for (c = 0; c<chunk_size; c++)
{  
    NLSE2D_STEP(ktotr,ktoti,Uoldr,Uoldi,V,Dr,Di,s1,s2,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);
    add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N,M);          
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s1,s2,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);
    add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N,M);   
    add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N,M);   
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s1,s2,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);  
    add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N,M);   
    add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N,M);  
    NLSE2D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s1,s2,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);       
    add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0,N,M);   
    add_matrix(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N,M);  

}/*End of c chunk_size*/

Unewr = mxGetPr(plhs[0]);
Unewi = mxGetPi(plhs[0]);
/*Copy 2D array into vector*/
/*Copy 2D array into vector*/
for(i=0;i<N;i++){
    for(j=0;j<M;j++){
        Unewr[M*i+j] = Uoldr[i][j];
        Unewi[M*i+j] = Uoldi[i][j];
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
}


free(Uoldr);  
free(Uoldi);
free(Utmpr);  
free(Utmpi);
free(ktmpr);  
free(ktmpi);
free(ktotr);  
free(ktoti); 
free(V);
free(Dr);
free(Di);

if(!mxIsComplex(prhs[0])){
	free(vUoldi);
}

}

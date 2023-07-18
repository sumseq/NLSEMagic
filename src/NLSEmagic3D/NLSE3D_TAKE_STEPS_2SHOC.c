/*----------------------------
NLSE3D_TAKE_STEPS_2SHOC.c
Program to integrate a chunk of time steps of the 3D Nonlinear Shrodinger Equation
i*Ut + a*(Uxx + Uyy + Uzz) - V(r)*U + s*|U|^2*U = 0
using RK4 + 2SHOC in double precision.

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

void add_cube(double*** Ar,    double*** Ai,
              double*** Br,    double*** Bi,
              double*** Cr,    double*** Ci,
              double K, int L, int N, int M)
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

void NLSE3D_STEP(double*** NLSFr, double*** NLSFi, 
                 double*** Utmpr, double*** Utmpi, 
                 double*** V, double*** Dr, double*** Di,
                 double s, double a, 
                 double a_6h2, double a_12, double lh2, 
                 double l_a,        
                 int BC, int L, int N, int M){

int i,j,k;	
double A,OM, Nb, Nb1;
int i1,j1;

for(i=1;i<L-1;i++)
{       
  for(j=1;j<N-1;j++)
  {
      for(k=1;k<M-1;k++)
      {   
         Dr[i][j][k] =  lh2*(Utmpr[i+1][j][k] + Utmpr[i-1][j][k] +
                             Utmpr[i][j+1][k] + Utmpr[i][j-1][k] +
                             Utmpr[i][j][k+1] + Utmpr[i][j][k-1] 
                                            - 6*Utmpr[i][j][k]);
         Di[i][j][k] =  lh2*(Utmpi[i+1][j][k] + Utmpi[i-1][j][k] +
                             Utmpi[i][j+1][k] + Utmpi[i][j-1][k] +
                             Utmpi[i][j][k+1] + Utmpi[i][j][k-1] 
                                            - 6*Utmpi[i][j][k]);    
      }  
  }    
}
/*Boundary conditions:*/
switch (BC){
    case 1: /*Dirichlet*/
    for(i=0;i<L;i++){
        for(j=0;j<N;j++){
           Dr[i][j][0]     = l_a*(V[i][j][0]   - s*(Utmpr[i][j][0]*Utmpr[i][j][0]     + Utmpi[i][j][0]*Utmpi[i][j][0]))*Utmpr[i][j][0]; 
           Di[i][j][0]     = l_a*(V[i][j][0]   - s*(Utmpr[i][j][0]*Utmpr[i][j][0]     + Utmpi[i][j][0]*Utmpi[i][j][0]))*Utmpi[i][j][0]; 
           Dr[i][j][M-1]   = l_a*(V[i][j][M-1] - s*(Utmpr[i][j][M-1]*Utmpr[i][j][M-1] + Utmpi[i][j][M-1]*Utmpi[i][j][M-1]))*Utmpr[i][j][M-1]; 
           Di[i][j][M-1]   = l_a*(V[i][j][M-1] - s*(Utmpr[i][j][M-1]*Utmpr[i][j][M-1] + Utmpi[i][j][M-1]*Utmpi[i][j][M-1]))*Utmpi[i][j][M-1]; 
        }
    }
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
           Dr[i][0][k]     = l_a*(V[i][0][k]   - s*(Utmpr[i][0][k]*Utmpr[i][0][k]     + Utmpi[i][0][k]*Utmpi[i][0][k]))*Utmpr[i][0][k];  
           Di[i][0][k]     = l_a*(V[i][0][k]   - s*(Utmpr[i][0][k]*Utmpr[i][0][k]     + Utmpi[i][0][k]*Utmpi[i][0][k]))*Utmpi[i][0][k];
           Dr[i][N-1][k]   = l_a*(V[i][N-1][k] - s*(Utmpr[i][N-1][k]*Utmpr[i][N-1][k] + Utmpi[i][N-1][k]*Utmpi[i][N-1][k]))*Utmpr[i][N-1][k];   
           Di[i][N-1][k]   = l_a*(V[i][N-1][k] - s*(Utmpr[i][N-1][k]*Utmpr[i][N-1][k] + Utmpi[i][N-1][k]*Utmpi[i][N-1][k]))*Utmpi[i][N-1][k];   
        }
    }
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){
           Dr[0][j][k]     = l_a*(V[0][j][k]   - s*(Utmpr[0][j][k]*Utmpr[0][j][k]     + Utmpi[0][j][k]*Utmpi[0][j][k]))*Utmpr[0][j][k];  
           Di[0][j][k]     = l_a*(V[0][j][k]   - s*(Utmpr[0][j][k]*Utmpr[0][j][k]     + Utmpi[0][j][k]*Utmpi[0][j][k]))*Utmpi[0][j][k];  
           Dr[L-1][j][k]   = l_a*(V[L-1][j][k] - s*(Utmpr[L-1][j][k]*Utmpr[L-1][j][k] + Utmpi[L-1][j][k]*Utmpi[L-1][j][k]))*Utmpr[L-1][j][k];   
           Di[L-1][j][k]   = l_a*(V[L-1][j][k] - s*(Utmpr[L-1][j][k]*Utmpr[L-1][j][k] + Utmpi[L-1][j][k]*Utmpi[L-1][j][k]))*Utmpi[L-1][j][k];   
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

        A   = (Dr[i1][j1][1]*Utmpr[i1][j1][1] + Di[i1][j1][1]*Utmpi[i1][j1][1])/
              (Utmpr[i1][j1][1]*Utmpr[i1][j1][1] + Utmpi[i1][j1][1]*Utmpi[i1][j1][1]);       
        Nb  = s*(Utmpr[i][j][0]*Utmpr[i][j][0] + Utmpi[i][j][0]*Utmpi[i][j][0]) - V[i][j][0];
        Nb1 = s*(Utmpr[i1][j1][1]*Utmpr[i1][j1][1] + Utmpi[i1][j1][1]*Utmpi[i1][j1][1]) - V[i1][j1][1];	   
		
        Dr[i][j][0]  = (A + l_a*(Nb1-Nb))*Utmpr[i][j][0];
        Di[i][j][0]  = (A + l_a*(Nb1-Nb))*Utmpi[i][j][0]; 

        A   = (Dr[i1][j1][M-2]*Utmpr[i1][j1][M-2] + Di[i1][j1][M-2]*Utmpi[i1][j1][M-2])/
              (Utmpr[i1][j1][M-2]*Utmpr[i1][j1][M-2] + Utmpi[i1][j1][M-2]*Utmpi[i1][j1][M-2]);       
        Nb  = s*(Utmpr[i][j][M-1]*Utmpr[i][j][M-1] + Utmpi[i][j][M-1]*Utmpi[i][j][M-1]) - V[i][j][M-1];
        Nb1 = s*(Utmpr[i1][j1][M-2]*Utmpr[i1][j1][M-2] + Utmpi[i1][j1][M-2]*Utmpi[i1][j1][M-2]) - V[i1][j1][M-2];	   
		
        Dr[i][j][M-1]  = (A + l_a*(Nb1-Nb))*Utmpr[i][j][M-1];
        Di[i][j][M-1]  = (A + l_a*(Nb1-Nb))*Utmpi[i][j][M-1]; 
        }/*j*/
    }/*i*/
    /*Now do L-M planes for j=0 and j=N-1
      not repeating edges and corners above*/
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){                    
        i1 = i;        
        if(i==0)     i1 = 1;
        if(i==L-1)   i1 = L-2;

        A   = (Dr[i1][1][k]*Utmpr[i1][1][k] + Di[i1][1][k]*Utmpi[i1][1][k])/
              (Utmpr[i1][1][k]*Utmpr[i1][1][k] + Utmpi[i1][1][k]*Utmpi[i1][1][k]);       
        Nb  = s*(Utmpr[i][0][k]*Utmpr[i][0][k] + Utmpi[i][0][k]*Utmpi[i][0][k]) - V[i][0][k];
        Nb1 = s*(Utmpr[i1][1][k]*Utmpr[i1][1][k] + Utmpi[i1][1][k]*Utmpi[i1][1][k]) - V[i1][1][k];	   
		
        Dr[i][0][k]  = (A + l_a*(Nb1-Nb))*Utmpr[i][0][k];
        Di[i][0][k]  = (A + l_a*(Nb1-Nb))*Utmpi[i][0][k]; 

        A   = (Dr[i1][N-2][k]*Utmpr[i1][N-2][k] + Di[i1][N-2][k]*Utmpi[i1][N-2][k])/
              (Utmpr[i1][N-2][k]*Utmpr[i1][N-2][k] + Utmpi[i1][N-2][k]*Utmpi[i1][N-2][k]);       
        Nb  = s*(Utmpr[i][N-1][k]*Utmpr[i][N-1][k] + Utmpi[i][N-1][k]*Utmpi[i][N-1][k]) - V[i][N-1][k];
        Nb1 = s*(Utmpr[i1][N-2][k]*Utmpr[i1][N-2][k] + Utmpi[i1][N-2][k]*Utmpi[i1][N-2][k]) - V[i1][N-2][k];	   
		
        Dr[i][N-1][k]  = (A + l_a*(Nb1-Nb))*Utmpr[i][N-1][k];
        Di[i][N-1][k]  = (A + l_a*(Nb1-Nb))*Utmpi[i][N-1][k]; 
        }/*k*/
    }/*i*/
    /*Now do N-M planes for i=0 and i=L-1
      no more edges or corners to worry about*/
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){

        A   = (Dr[1][j][k]*Utmpr[1][j][k] + Di[1][j][k]*Utmpi[1][j][k])/
              (Utmpr[1][j][k]*Utmpr[1][j][k] + Utmpi[1][j][k]*Utmpi[1][j][k]);       
        Nb  = s*(Utmpr[0][j][k]*Utmpr[0][j][k] + Utmpi[0][j][k]*Utmpi[0][j][k]) - V[0][j][k];
        Nb1 = s*(Utmpr[1][j][k]*Utmpr[1][j][k] + Utmpi[1][j][k]*Utmpi[1][j][k]) - V[1][j][k];	   
		
        Dr[0][j][k]  = (A + l_a*(Nb1-Nb))*Utmpr[0][j][k];
        Di[0][j][k]  = (A + l_a*(Nb1-Nb))*Utmpi[0][j][k];

        A   = (Dr[L-2][j][k]*Utmpr[L-2][j][k] + Di[L-2][j][k]*Utmpi[L-2][j][k])/
              (Utmpr[L-2][j][k]*Utmpr[L-2][j][k] + Utmpi[L-2][j][k]*Utmpi[L-2][j][k]);       
        Nb  = s*(Utmpr[L-1][j][k]*Utmpr[L-1][j][k] + Utmpi[L-1][j][k]*Utmpi[L-1][j][k]) - V[L-1][j][k];
        Nb1 = s*(Utmpr[L-2][j][k]*Utmpr[L-2][j][k] + Utmpi[L-2][j][k]*Utmpi[L-2][j][k]) - V[L-2][j][k];	   
		
        Dr[L-1][j][k]  = (A + l_a*(Nb1-Nb))*Utmpr[L-1][j][k];
        Di[L-1][j][k]  = (A + l_a*(Nb1-Nb))*Utmpi[L-1][j][k];
        }
    } 
    break;   
    case 3:
    for(i=0;i<L;i++){
        for(j=0;j<N;j++){
           Dr[i][j][0]     = 0.0;
           Di[i][j][0]     = 0.0;
           Dr[i][j][M-1]   = 0.0;
           Di[i][j][M-1]   = 0.0;
        }
    }
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
           Dr[i][0][k]     = 0.0;
           Di[i][0][k]     = 0.0;
           Dr[i][N-1][k]   = 0.0;
           Di[i][N-1][k]   = 0.0;
        }
    }
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){
           Dr[0][j][k]     = 0.0;
           Di[0][j][k]     = 0.0;
           Dr[L-1][j][k]   = 0.0;
           Di[L-1][j][k]   = 0.0;
        }
    }    
    break;
    default:
    for(i=0;i<L;i++){
        for(j=0;j<N;j++){
           Dr[i][j][0]     = 0.0;
           Di[i][j][0]     = 0.0;
           Dr[i][j][M-1]   = 0.0;
           Di[i][j][M-1]   = 0.0;
        }
    }
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
           Dr[i][0][k]     = 0.0;
           Di[i][0][k]     = 0.0;
           Dr[i][N-1][k]   = 0.0;
           Di[i][N-1][k]   = 0.0;
        }
    }
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){
           Dr[0][j][k]     = 0.0;
           Di[0][j][k]     = 0.0;
           Dr[L-1][j][k]   = 0.0;
           Di[L-1][j][k]   = 0.0;
        }
    }    
    break;
}/*BC Switch*/  

/*Now compute 2SHOC*/
for (i=1;i<L-1;i++)
{       
  for(j=1;j<N-1;j++)
  {
    for(k=1;k<M-1;k++)
	  {
        NLSFr[i][j][k] =  a_12*(Di[i+1][j][k] + Di[i-1][j][k] +
                                Di[i][j+1][k] + Di[i][j-1][k] +
                                Di[i][j][k+1] + Di[i][j][k-1] -
                             10*Di[i][j][k])
          - a_6h2*(Utmpi[i+1][j+1][k] + Utmpi[i][j+1][k+1] + Utmpi[i+1][j][k+1] +
                   Utmpi[i-1][j-1][k] + Utmpi[i][j-1][k-1] + Utmpi[i-1][j][k-1] +
                   Utmpi[i+1][j-1][k] + Utmpi[i][j+1][k-1] + Utmpi[i+1][j][k-1] +
                   Utmpi[i-1][j+1][k] + Utmpi[i][j-1][k+1] + Utmpi[i-1][j][k+1] -
                12*Utmpi[i][j][k]) 
         + (V[i][j][k] - s*(Utmpr[i][j][k]*Utmpr[i][j][k] + Utmpi[i][j][k]*Utmpi[i][j][k]))*Utmpi[i][j][k];  
   
       NLSFi[i][j][k] = -a_12*(Dr[i+1][j][k] + Dr[i-1][j][k] +
                               Dr[i][j+1][k] + Dr[i][j-1][k] +
                               Dr[i][j][k+1] + Dr[i][j][k-1] -
                            10*Dr[i][j][k])
          + a_6h2*(Utmpr[i+1][j+1][k] + Utmpr[i][j+1][k+1] + Utmpr[i+1][j][k+1] +
                   Utmpr[i-1][j-1][k] + Utmpr[i][j-1][k-1] + Utmpr[i-1][j][k-1] +
                   Utmpr[i+1][j-1][k] + Utmpr[i][j+1][k-1] + Utmpr[i+1][j][k-1] +
                   Utmpr[i-1][j+1][k] + Utmpr[i][j-1][k+1] + Utmpr[i-1][j][k+1] -
                12*Utmpr[i][j][k])    
	     - (V[i][j][k] - s*(Utmpr[i][j][k]*Utmpr[i][j][k] + Utmpi[i][j][k]*Utmpi[i][j][k]))*Utmpr[i][j][k];
    }    
  }
}

/*Boundary conditions:*/
switch (BC){
    case 1:
    for(i=0;i<L;i++){
        for(j=0;j<N;j++){
           NLSFr[i][j][0]     = 0.0; 
           NLSFi[i][j][0]     = 0.0;
           NLSFr[i][j][M-1]   = 0.0; 
           NLSFi[i][j][M-1]   = 0.0;
        }
    }
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
           NLSFr[i][0][k]     = 0.0; 
           NLSFi[i][0][k]     = 0.0;
           NLSFr[i][N-1][k]   = 0.0; 
           NLSFi[i][N-1][k]   = 0.0;
        }
    }
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){
           NLSFr[0][j][k]     = 0.0; 
           NLSFi[0][j][k]     = 0.0;
           NLSFr[L-1][j][k]   = 0.0; 
           NLSFi[L-1][j][k]   = 0.0;
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
    case 3:
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
           NLSFr[i][j][0]     = 0.0; 
           NLSFi[i][j][0]     = 0.0;
           NLSFr[i][j][M-1]   = 0.0; 
           NLSFi[i][j][M-1]   = 0.0;
        }
    }
    for(i=0;i<L;i++){
        for(k=1;k<M-1;k++){
           NLSFr[i][0][k]     = 0.0; 
           NLSFi[i][0][k]     = 0.0;
           NLSFr[i][N-1][k]   = 0.0; 
           NLSFi[i][N-1][k]   = 0.0;
        }
    }
    for(j=1;j<N-1;j++){
        for(k=1;k<M-1;k++){
           NLSFr[0][j][k]     = 0.0; 
           NLSFi[0][j][k]     = 0.0;
           NLSFr[L-1][j][k]   = 0.0; 
           NLSFi[L-1][j][k]   = 0.0;
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
double h2, a, s, K, K2, K6, ah2, lh2, l_a, a_6h2, a_12;
double *vUoldr, *vUoldi,  *vUnewr, *vUnewi, *vV;
double ***Uoldr, ***Uoldi, ***V;
double ***Utmpr, ***Utmpi, ***Dr, ***Di;
double ***ktmpr, ***ktmpi, ***ktotr, ***ktoti;

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
vV     = mxGetPr(prhs[1]);

Uoldr   = (double ***) malloc(sizeof(double**)*L);
Uoldi   = (double ***) malloc(sizeof(double**)*L);
V       = (double ***) malloc(sizeof(double**)*L);
Utmpr   = (double ***) malloc(sizeof(double**)*L);
Utmpi   = (double ***) malloc(sizeof(double**)*L);
Dr      = (double ***) malloc(sizeof(double**)*L);
Di      = (double ***) malloc(sizeof(double**)*L);
ktmpr   = (double ***) malloc(sizeof(double**)*L);
ktmpi   = (double ***) malloc(sizeof(double**)*L);
ktotr   = (double ***) malloc(sizeof(double**)*L);
ktoti   = (double ***) malloc(sizeof(double**)*L);

for (i=0;i<L;i++){   
    V[i]     = (double **) malloc(sizeof(double*)*N);
    Uoldr[i] = (double **) malloc(sizeof(double*)*N);
    Uoldi[i] = (double **) malloc(sizeof(double*)*N);
    Utmpr[i] = (double **) malloc(sizeof(double*)*N);
    Utmpi[i] = (double **) malloc(sizeof(double*)*N);
    Dr[i]    = (double **) malloc(sizeof(double*)*N);
    Di[i]    = (double **) malloc(sizeof(double*)*N);
    ktmpr[i] = (double **) malloc(sizeof(double*)*N);
    ktmpi[i] = (double **) malloc(sizeof(double*)*N);
    ktotr[i] = (double **) malloc(sizeof(double*)*N);
    ktoti[i] = (double **) malloc(sizeof(double*)*N);  
    for (j=0; j<N; j++){
         V[i][j]     = vV     + N*M*i + M*j;
         Uoldr[i][j] = vUoldr + N*M*i + M*j;
         Uoldi[i][j] = vUoldi + N*M*i + M*j;
         Utmpr[i][j] = (double *) malloc(sizeof(double)*M);
         Utmpi[i][j] = (double *) malloc(sizeof(double)*M);
         Dr[i][j]    = (double *) malloc(sizeof(double)*M);
         Di[i][j]    = (double *) malloc(sizeof(double)*M);
         ktmpr[i][j] = (double *) malloc(sizeof(double)*M);
         ktmpi[i][j] = (double *) malloc(sizeof(double)*M);
         ktotr[i][j] = (double *) malloc(sizeof(double)*M);
         ktoti[i][j] = (double *) malloc(sizeof(double)*M);  
    }
}

 
s          = (double)mxGetScalar(prhs[2]);
a          = (double)mxGetScalar(prhs[3]);
h2         = (double)mxGetScalar(prhs[4]);
BC         = (int)mxGetScalar(prhs[5]);
chunk_size = (int)mxGetScalar(prhs[6]);
K          = (double)mxGetScalar(prhs[7]);

lh2    = 1.0/h2;
a_6h2  = a/(6.0*h2);
a_12   = a/12.0;
l_a    = 1.0/a;
K2     = K/2.0;
K6     = K/6.0;


for (c = 0; c<chunk_size; c++)
{   
    NLSE3D_STEP(ktotr, ktoti,Uoldr,Uoldi,V,Dr,Di,s,a,a_6h2,a_12,lh2,l_a,BC,L,N,M);
    add_cube(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,K2,L,N,M);          
    NLSE3D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a_6h2,a_12,lh2,l_a,BC,L,N,M);
    add_cube(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,L,N,M);   
    add_cube(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,K2,L,N,M);   
    NLSE3D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a_6h2,a_12,lh2,l_a,BC,L,N,M);  
    add_cube(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,L,N,M);   
    add_cube(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,K,L,N,M);  
    NLSE3D_STEP(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a_6h2,a_12,lh2,l_a,BC,L,N,M);       
    add_cube(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0,L,N,M);   
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
           vUnewr[N*M*i + M*j + k] = Uoldr[i][j][k];
           vUnewi[N*M*i + M*j + k] = Uoldi[i][j][k];
        }
    }
}

    
/*Free up memory:*/
for (i=0;i<L;i++){   
    for(j=0;j<N;j++){
       free(Utmpr[i][j]);
       free(Utmpi[i][j]);
       free(Dr[i][j]);
       free(Di[i][j]);
       free(ktmpr[i][j]);
       free(ktmpi[i][j]);
       free(ktotr[i][j]);
       free(ktoti[i][j]);
    }
}
for (i=0; i<L; i++){          
       free(Utmpr[i]);
       free(Utmpi[i]);
       free(Dr[i]);
       free(Di[i]);
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
free(Dr);     free(Di);
free(ktmpr);  free(ktmpi);
free(ktotr);  free(ktoti); 
free(V);

if(!mxIsComplex(prhs[0])){
	free(vUoldi);
}

}

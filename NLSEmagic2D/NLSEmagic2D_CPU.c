#include <stdlib.h>

void add_matrix(double** Ar, double** Ai, double** Br, double** Bi, double** Cr, double** Ci, double k, int N, int M){
	int i,j;
	for(i=0;i<N;i++){
	   for(j=0;j<M;j++){
    	Ar[i][j]  = Br[i][j] + k*Cr[i][j];
    	Ai[i][j]  = Bi[i][j] + k*Ci[i][j];
	   }
    }
}

void add_matrix_F(float** Ar, float** Ai, float** Br, float** Bi, float** Cr, float** Ci, float k, int N, int M){
	int i,j;
	for(i=0;i<N;i++){
	   for(j=0;j<M;j++){
    	Ar[i][j]  = Br[i][j] + k*Cr[i][j];
    	Ai[i][j]  = Bi[i][j] + k*Ci[i][j];
	   }
    }
}

void compute_F_CD(double** NLSFr, double** NLSFi, double** Utmpr, double** Utmpi, double** V, double s, double ah2, 
                  int BC, int N, int M){

int i,j,k,i1,j1;	
double OM;
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

void compute_F_CD_F(float** NLSFr, float** NLSFi, float** Utmpr, float** Utmpi, 
                    float** V, float s, float ah2, int BC, int N, int M){

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

void compute_F_2SHOC(double** NLSFr, double** NLSFi, double** Utmpr, double** Utmpi, 
                     double** V, double** Dr, double** Di, double s, double a, 
                     double a1_6h2, double a1_12, double lh2, double l_a, 
                     int BC, int N, int M){

int i,j,k,i1,j1;	
double A,Nb1,Nb,OM;
int msd_x0[4] = {0,0,  N-1,N-1};
int msd_y0[4] = {0,M-1,0,  M-1};
int msd_x1[4] = {1,1,  N-2,N-2};
int msd_y1[4] = {1,M-2,1,  M-2};

/*Compute second-order Laplacian*/
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
    case 3: /*Lap0*/
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

/*Now compute Ut*/
for (i=1;i<N-1;i++)
{       
  for(j=1;j<M-1;j++)
  {

      NLSFr[i][j] = -a*Di[i][j] + a1_12*(Di[i+1][j] + Di[i-1][j] +
                                         Di[i][j+1] + Di[i][j-1]) 
                    - a1_6h2*(Utmpi[i+1][j+1] + Utmpi[i+1][j-1]  
                    - 4*Utmpi[i][j] + Utmpi[i-1][j+1] + Utmpi[i-1][j-1])
                    - (s*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]) 
                    - V[i][j])*Utmpi[i][j];
    
      NLSFi[i][j] =  a*Dr[i][j] - a1_12*(Dr[i+1][j] + Dr[i-1][j] + 
                                         Dr[i][j+1] + Dr[i][j-1])
                   + a1_6h2*(Utmpr[i+1][j+1] + Utmpr[i+1][j-1]  
                   - 4*Utmpr[i][j] + Utmpr[i-1][j+1] + Utmpr[i-1][j-1]) 
                   + (s*(Utmpr[i][j]*Utmpr[i][j] + Utmpi[i][j]*Utmpi[i][j]) 
                   - V[i][j])*Utmpr[i][j];
  }    
}
/*Boundary conditions:*/
switch (BC){
    case 1: /*Dirichlet*/
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
    case 3: /*lap0*/
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
 
void compute_F_2SHOC_F(float** NLSFr, float** NLSFi, float** Utmpr, float** Utmpi, 
                       float** V, float** Dr, float** Di, float s, float a, 
                       float a1_6h2, float a1_12, float lh2, float l_a,
                       int BC, int N, int M){

int i,j,k,i1,j1;	
float A,Nb1,Nb,OM;
int msd_x0[4] = {0,0,  N-1,N-1};
int msd_y0[4] = {0,M-1,0,  M-1};
int msd_x1[4] = {1,1,  N-2,N-2};
int msd_y1[4] = {1,M-2,1,  M-2};

/*First compute Laplacian*/
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

extern "C" void NLSE2D_TAKE_STEPS(double* vUoldr, double* vUoldi, double* vV,
                                  double* vUnewr, double* vUnewi,
                                  double s, double a, double h2, int BC,
                                  int chunk_size, double k, int N, int M, int method)
{
	double lh2, a1_6h2, a1_12, k2, k6, l_a, ah2;
    double **ktotr, **ktoti, **ktmpr, **ktmpi, **V;
    double **Uoldr, **Uoldi, **Utmpr, **Utmpi, **Dr, **Di;
    int i,j;

    /*Pre-compute scalars:*/
    l_a    = 1.0/a;
    lh2    = 1.0/h2;
    k2     = k/2.0;
    k6     = k/6.0;
    a1_6h2 = a*(1.0/(6.0*h2));
    a1_12  = a*(1.0/12.0);  
	ah2    = a/h2;
  
    /*Allocate 2D arrays*/
    Uoldr = (double **) malloc(sizeof(double*)*N);
    Uoldi = (double **) malloc(sizeof(double*)*N);
    V     = (double **) malloc(sizeof(double*)*N);
    Utmpr = (double **) malloc(sizeof(double*)*N);
    Utmpi = (double **) malloc(sizeof(double*)*N);
    ktmpr = (double **) malloc(sizeof(double*)*N);
    ktmpi = (double **) malloc(sizeof(double*)*N);
    ktotr = (double **) malloc(sizeof(double*)*N);
    ktoti = (double **) malloc(sizeof(double*)*N);	
	if(method==2){
      Dr  = (double **) malloc(sizeof(double*)*N);
      Di  = (double **) malloc(sizeof(double*)*N);
    }	

    for (i=0;i<N;i++){   
      V[i]     = vV+M*i;            
      Uoldr[i] = vUoldr+M*i;   
      Uoldi[i] = vUoldi+M*i; 
      Utmpr[i] = (double *) malloc(sizeof(double)*M);
      Utmpi[i] = (double *) malloc(sizeof(double)*M);
      ktmpr[i] = (double *) malloc(sizeof(double)*M);
      ktmpi[i] = (double *) malloc(sizeof(double)*M);
      ktotr[i] = (double *) malloc(sizeof(double)*M);
      ktoti[i] = (double *) malloc(sizeof(double)*M); 
	  if(method==2){
	     Dr[i] = (double *) malloc(sizeof(double)*M);
         Di[i] = (double *) malloc(sizeof(double)*M);
	  }
    }    
    /*Compute chunk of time steps using RK4*/
    if (method==1){	
      for (j = 0; j<chunk_size; j++){   
        compute_F_CD(ktotr,ktoti,Uoldr,Uoldi,V,s,ah2,BC,N,M);
        add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N,M);          
        compute_F_CD(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N,M);
        add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N,M);   
        add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N,M);   
        compute_F_CD(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N,M);  
        add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N,M);   
        add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N,M);  
        compute_F_CD(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N,M);       
        add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0,N,M);   
        add_matrix(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N,M);  
      }/*End of j chunk_size*/
    }else if (method==2)
    {
      for (j = 0; j<chunk_size; j++)
      {   
        compute_F_2SHOC(ktotr,ktoti,Uoldr,Uoldi,V,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);
        add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N,M);          
        compute_F_2SHOC(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);
        add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N,M);   
        add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N,M);   
        compute_F_2SHOC(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);  
        add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N,M);   
        add_matrix(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N,M);  
        compute_F_2SHOC(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);       
        add_matrix(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0,N,M);   
        add_matrix(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N,M);  
      }/*End of j chunk_size*/
	}
	
    /*Copy 2D array into result vector*/
    for(i=0;i<N;i++){
      for(j=0;j<M;j++){
        vUnewr[M*i+j] = Uoldr[i][j];
        vUnewi[M*i+j] = Uoldi[i][j];
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
      if (method==2){free(Dr[i]);free(Di[i]);}
    }
    free(Uoldr);  free(Uoldi);
    free(Utmpr);  free(Utmpi);
    free(ktmpr);  free(ktmpi);
    free(ktotr);  free(ktoti); 
    free(V);
	if (method==2){free(Dr);free(Di);}
}

extern "C" void NLSE2D_TAKE_STEPS_F(float* vUoldr, float* vUoldi, float* vV,
                                    double* vUnewr, double* vUnewi,
                                    float s, float a, float h2, int BC,
                                    int chunk_size, float k, int N, int M, int method)
{
    float lh2, a1_6h2, a1_12, k2, k6, l_a, ah2;
    float **ktotr, **ktoti, **ktmpr, **ktmpi, **V;
    float **Uoldr, **Uoldi, **Utmpr, **Utmpi, **Dr, **Di;
    int i,j;

    /*Pre-compute scalars:*/
    l_a    = 1.0/a;
    lh2    = 1.0/h2;
    k2     = k/2.0;
    k6     = k/6.0;
    a1_6h2 = a*(1.0/(6.0*h2));
    a1_12  = a*(1.0/12.0);  
	ah2    = a/h2;  
  
    /*Allocate 2D arrays*/
    Uoldr = (float **) malloc(sizeof(float*)*N);
    Uoldi = (float **) malloc(sizeof(float*)*N);
    V     = (float **) malloc(sizeof(float*)*N);
    Utmpr = (float **) malloc(sizeof(float*)*N);
    Utmpi = (float **) malloc(sizeof(float*)*N);
    ktmpr = (float **) malloc(sizeof(float*)*N);
    ktmpi = (float **) malloc(sizeof(float*)*N);
    ktotr = (float **) malloc(sizeof(float*)*N);
    ktoti = (float **) malloc(sizeof(float*)*N);	
	if(method==2){
      Dr  = (float **) malloc(sizeof(float*)*N);
      Di  = (float **) malloc(sizeof(float*)*N);
    }	

    for (i=0;i<N;i++){   
      V[i]     = vV+M*i;            
      Uoldr[i] = vUoldr+M*i;   
      Uoldi[i] = vUoldi+M*i; 
      Utmpr[i] = (float *) malloc(sizeof(float)*M);
      Utmpi[i] = (float *) malloc(sizeof(float)*M);
      ktmpr[i] = (float *) malloc(sizeof(float)*M);
      ktmpi[i] = (float *) malloc(sizeof(float)*M);
      ktotr[i] = (float *) malloc(sizeof(float)*M);
      ktoti[i] = (float *) malloc(sizeof(float)*M); 
	  if(method==2){
	     Dr[i] = (float *) malloc(sizeof(float)*M);
         Di[i] = (float *) malloc(sizeof(float)*M);
	  }
    }    
    /*Compute chunk of time steps using RK4*/
    if (method==1){	
      for (j = 0; j<chunk_size; j++){   
        compute_F_CD_F(ktotr,ktoti,Uoldr,Uoldi,V,s,ah2,BC,N,M);
        add_matrix_F(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N,M);          
        compute_F_CD_F(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N,M);
        add_matrix_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N,M);   
        add_matrix_F(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N,M);   
        compute_F_CD_F(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N,M);  
        add_matrix_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N,M);   
        add_matrix_F(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N,M);  
        compute_F_CD_F(ktmpr, ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N,M);       
        add_matrix_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0f,N,M);   
        add_matrix_F(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N,M);  
      }/*End of j chunk_size*/
    }else if (method==2)
    {
      for (j = 0; j<chunk_size; j++)
      {   
        compute_F_2SHOC_F(ktotr,ktoti,Uoldr,Uoldi,V,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);
        add_matrix_F(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N,M);          
        compute_F_2SHOC_F(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);
        add_matrix_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N,M);   
        add_matrix_F(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N,M);   
        compute_F_2SHOC_F(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);  
        add_matrix_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N,M);   
        add_matrix_F(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N,M);  
        compute_F_2SHOC_F(ktmpr, ktmpi,Utmpr,Utmpi,V,Dr,Di,s,a,a1_6h2,a1_12,lh2,l_a,BC,N,M);       
        add_matrix_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0f,N,M);   
        add_matrix_F(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N,M);  
      }/*End of j chunk_size*/
	}
	
    /*Copy 2D array into result vector*/
    for(i=0;i<N;i++){
      for(j=0;j<M;j++){
        vUnewr[M*i+j] = (double)Uoldr[i][j];
        vUnewi[M*i+j] = (double)Uoldi[i][j];
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
      if (method==2){free(Dr[i]);free(Di[i]);}
    }
    free(Uoldr);  free(Uoldi);
    free(Utmpr);  free(Utmpi);
    free(ktmpr);  free(ktmpi);
    free(ktotr);  free(ktoti); 
    free(V);
	if (method==2){free(Dr);free(Di);}
}

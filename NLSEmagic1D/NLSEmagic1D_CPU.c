#include <stdlib.h>

void vec_add(double* Ar, double* Ai,double* Br, double* Bi,double* Cr, double* Ci,double k, int N){
	int i = 0;	
	for (i = 0; i < N; i++){
	  	Ar[i]  = Br[i] + k*Cr[i];
    	Ai[i]  = Bi[i] + k*Ci[i]; 
	}
}

void vec_add_F(float* Ar, float* Ai,float* Br, float* Bi,float* Cr, float* Ci,float k, int N){
	int i = 0;	
	for (i = 0; i < N; i++){
	  	Ar[i]  = Br[i] + k*Cr[i];
    	Ai[i]  = Bi[i] + k*Ci[i]; 
	}
}

void compute_F_CD(double* NLSFr, double* NLSFi, double* Utmpr, double* Utmpi, 
                  double* V, double s, double ah2, int BC, int N){

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

void compute_F_CD_F(float* NLSFr, float* NLSFi, float* Utmpr, float* Utmpi, 
                    float* V, float s, float ah2, int BC, int N){

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

void compute_F_2SHOC(double* NLSFr, double* NLSFi, double* Utmpr, double* Utmpi, 
			         double* dx2r, double* dx2i, double* V, double s, 
                     double  a, double l_a, double a76, double a112, 
                     double  lh2, int BC, int N){

int i;
double OM,A,Nb,Nb1;

/*Compute second-order Laplacian*/
for (i = 1; i < N-1; i++)
{ 
	dx2r[i] =  (Utmpr[i+1] - 2*Utmpr[i] + Utmpr[i-1])*lh2; 
	dx2i[i] =  (Utmpi[i+1] - 2*Utmpi[i] + Utmpi[i-1])*lh2;
}
/*Boundary Conditions*/
switch (BC){
    case 1: /*Dirichlet*/
       dx2r[0]   = l_a*(V[0]   - s*(Utmpr[0]*Utmpr[0]     + Utmpi[0]*Utmpi[0]))*Utmpr[0];
	   dx2i[0]   = l_a*(V[0]   - s*(Utmpr[0]*Utmpr[0]     + Utmpi[0]*Utmpi[0]))*Utmpi[0]; 
	   dx2r[N-1] = l_a*(V[N-1] - s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]))*Utmpr[N-1]; 
	   dx2i[N-1] = l_a*(V[N-1] - s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]))*Utmpi[N-1]; 
       break;
    case 2: /*MSD*/
       A   =    (dx2r[1]*Utmpr[1] + dx2i[1]*Utmpi[1])/
               (Utmpr[1]*Utmpr[1] + Utmpi[1]*Utmpi[1]);       
       Nb  = s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0];
       Nb1 = s*(Utmpr[1]*Utmpr[1] + Utmpi[1]*Utmpi[1]) - V[1];	   
		
	   dx2r[0]  = (A + l_a*(Nb1-Nb))*Utmpr[0];
       dx2i[0]  = (A + l_a*(Nb1-Nb))*Utmpi[0];    
      
	   A   =   (dx2r[N-2]*Utmpr[N-2]  + dx2i[N-2]*Utmpi[N-2])/
               (Utmpr[N-2]*Utmpr[N-2] + Utmpi[N-2]*Utmpi[N-2]);       
       Nb  = s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1];
       Nb1 = s*(Utmpr[N-2]*Utmpr[N-2] + Utmpi[N-2]*Utmpi[N-2]) - V[N-2];	   
		
	   dx2r[N-1]  = (A + l_a*(Nb1-Nb))*Utmpr[N-1];
       dx2i[N-1]  = (A + l_a*(Nb1-Nb))*Utmpi[N-1];    
       break;
    case 3:  /*Lap0*/
       dx2r[0]   = 0.0; 
	   dx2i[0]   = 0.0; 
	   dx2r[N-1] = 0.0; 
	   dx2i[N-1] = 0.0;
       break;
    case 4: /*1-sided*/
       dx2r[0]   = (-Utmpr[3]   + 4*Utmpr[2]   - 5*Utmpr[1]   + 2*Utmpr[0])*lh2;
       dx2i[0]   = (-Utmpi[3]   + 4*Utmpi[2]   - 5*Utmpi[1]   + 2*Utmpi[0])*lh2;    
       dx2r[N-1] = (-Utmpr[N-4] + 4*Utmpr[N-3] - 5*Utmpr[N-2] + 2*Utmpr[N-1])*lh2;                     
       dx2i[N-1] = (-Utmpi[N-4] + 4*Utmpi[N-3] - 5*Utmpi[N-2] + 2*Utmpi[N-1])*lh2;    
       break;
    default:
 	   dx2r[0]   = 0.0; 
	   dx2i[0]   = 0.0; 
	   dx2r[N-1] = 0.0; 
	   dx2i[N-1] = 0.0;
       break;
}/*BC Switch*/


/* Now compute Ut: */
for (i = 1; i < N-1; i++)
{   
  NLSFr[i] = a112*(dx2i[i+1] + dx2i[i-1]) - a76*dx2i[i]
             - (s*(Utmpr[i]*Utmpr[i] + Utmpi[i]*Utmpi[i]) - V[i])*Utmpi[i];
  NLSFi[i] = a76*dx2r[i] - a112*(dx2r[i+1] + dx2r[i-1])
             + (s*(Utmpr[i]*Utmpr[i] + Utmpi[i]*Utmpi[i]) - V[i])*Utmpr[i];
}
/*Boundary Conditions*/
   switch (BC){
    case 1:  /*Dirichlet*/
      NLSFr[0]   = 0.0; 
	  NLSFi[0]   = 0.0; 
	  NLSFr[N-1] = 0.0; 
	  NLSFi[N-1] = 0.0;
      break;
    case 2:  /*MSD*/  
      OM =  (NLSFi[1]*Utmpr[1] -  NLSFr[1]*Utmpi[1])/
            (Utmpr[1]*Utmpr[1] + Utmpi[1]*Utmpi[1]);
                                        
      NLSFr[0]  = -OM*Utmpi[0];
      NLSFi[0]  =  OM*Utmpr[0];

      OM =  (NLSFi[N-2]*Utmpr[N-2] -  NLSFr[N-2]*Utmpi[N-2])/
            (Utmpr[N-2]*Utmpr[N-2] + Utmpi[N-2]*Utmpi[N-2]);
                                        
      NLSFr[N-1]  = -OM*Utmpi[N-1];
      NLSFi[N-1]  =  OM*Utmpr[N-1];      
      break;
    case 3: /*Lap0*/
      NLSFr[0]   = - (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpi[0];
	  NLSFi[0]   =   (s*(Utmpr[0]*Utmpr[0] + Utmpi[0]*Utmpi[0]) - V[0])*Utmpr[0];
	  NLSFr[N-1] = - (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpi[N-1];
	  NLSFi[N-1] =   (s*(Utmpr[N-1]*Utmpr[N-1] + Utmpi[N-1]*Utmpi[N-1]) - V[N-1])*Utmpr[N-1];
      break;
    case 4: /*1-sided*/    
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
      NLSFr[0]   = 0.0; 
	  NLSFi[0]   = 0.0; 
	  NLSFr[N-1] = 0.0; 
	  NLSFi[N-1] = 0.0;
      break;
   }/*BC Switch*/

}	      

void compute_F_2SHOC_F(float* NLSFr, float* NLSFi, float* Utmpr, float* Utmpi, 
			           float* dx2r, float* dx2i, float* V, float s, 
                       float a, float  l_a, float a76, float a112, 
                       float lh2, int BC, int N){

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

extern "C" void NLSE1D_TAKE_STEPS(double *Uoldr, double* Uoldi, double* V,
                                  double *Unewr, double* Unewi,
                                  double s, double a, double h2, int BC,
                                  int chunk_size, double k, int N, int method)
{
    double ah2,k2,k6,l_a,lh2,a76,a112;
    double *ktotr, *ktoti,*ktmpr,*ktmpi;
    double *Utmpr, *Utmpi, *dx2r, *dx2i;
    int j;

    /*Precompute scalars:*/
    ah2 = a/h2;
    k2  = k/2.0;
    k6  = k/6.0;
    l_a  = 1.0/a;
    lh2  = 1.0/h2;
    a76  = a*(7.0/6.0);
    a112 = a*(1.0/12.0);    
  
    Utmpr   = (double *) malloc(sizeof(double)*N);
    Utmpi   = (double *) malloc(sizeof(double)*N);
    ktmpr   = (double *) malloc(sizeof(double)*N);
    ktmpi   = (double *) malloc(sizeof(double)*N);
    ktotr   = (double *) malloc(sizeof(double)*N);
    ktoti   = (double *) malloc(sizeof(double)*N);
	if(method==2){
	  dx2r   = (double *) malloc(sizeof(double)*N);
      dx2i   = (double *) malloc(sizeof(double)*N);
	}

    /*Compute chunk of time steps using RK4*/
    if (method==1){	
      for (j = 0; j<chunk_size; j++)
      {   
      compute_F_CD(ktotr,ktoti,Uoldr,Uoldi,V,s,ah2,BC,N);
      vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N);            
      compute_F_CD(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);
      vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N);
      vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N); 
      compute_F_CD(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);    
      vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N);
      vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N);       
      compute_F_CD(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);         
      vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0,N);
      vec_add(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N);   
      }/*End of j chunk_size*/
    }else if (method==2)
    {
      for (j = 0; j<chunk_size; j++)
      {   
      compute_F_2SHOC(ktotr, ktoti,Uoldr,Uoldi,dx2r,dx2i,V,s,a,l_a,a76,a112,lh2,BC,N);
      vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N);          
      compute_F_2SHOC(ktmpr, ktmpi,Utmpr,Utmpi,dx2r,dx2i,V,s,a,l_a,a76,a112,lh2,BC,N);
      vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N);
      vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N); 
      compute_F_2SHOC(ktmpr, ktmpi,Utmpr,Utmpi,dx2r,dx2i,V,s,a,l_a,a76,a112,lh2,BC,N);    
      vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0,N);
      vec_add(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N);       
      compute_F_2SHOC(ktmpr, ktmpi,Utmpr,Utmpi,dx2r,dx2i,V,s,a,l_a,a76,a112,lh2,BC,N);          
      vec_add(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0,N);
      vec_add(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N);   
      }/*End of j chunk_size*/
	}
	
    for(j=0;j<N;j++){
      Unewr[j] = Uoldr[j];
      Unewi[j] = Uoldi[j];
    }

    /*Free up memory:*/
    free(Utmpr); free(Utmpi);
    free(ktmpr); free(ktmpi);
    free(ktotr); free(ktoti); 
	if (method==2){free(dx2r);free(dx2i);}
}

extern "C" void NLSE1D_TAKE_STEPS_F(float *Uoldr, float* Uoldi, float* V,
                                    float *Unewr, float* Unewi,
                                    float s, float a, float h2, int BC,
                                    int chunk_size, float k, int N, int method)
{
    float ah2,k2,k6,l_a,lh2,a76,a112;
    float *ktotr, *ktoti,*ktmpr,*ktmpi;
    float *Utmpr, *Utmpi, *dx2r, *dx2i;
    int j;

    /*Precompute scalars:*/
    ah2 = a/h2;
    k2  = k/2.0f;
    k6  = k/6.0f;
    l_a  = 1.0f/a;
    lh2  = 1.0f/h2;
    a76  = a*(7.0f/6.0f);
    a112 = a*(1.0f/12.0f);    
  
    Utmpr   = (float *) malloc(sizeof(float)*N);
    Utmpi   = (float *) malloc(sizeof(float)*N);
    ktmpr   = (float *) malloc(sizeof(float)*N);
    ktmpi   = (float *) malloc(sizeof(float)*N);
    ktotr   = (float *) malloc(sizeof(float)*N);
    ktoti   = (float *) malloc(sizeof(float)*N);
	if(method==2){
	  dx2r   = (float *) malloc(sizeof(float)*N);
      dx2i   = (float *) malloc(sizeof(float)*N);
	}

    /*Compute chunk of time steps using RK4*/
    if (method==1){	
      for (j = 0; j<chunk_size; j++)
      {   
      compute_F_CD_F(ktotr,ktoti,Uoldr,Uoldi,V,s,ah2,BC,N);
      vec_add_F(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N);            
      compute_F_CD_F(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);
      vec_add_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N);
      vec_add_F(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N); 
      compute_F_CD_F(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);    
      vec_add_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N);
      vec_add_F(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N);       
      compute_F_CD_F(ktmpr,ktmpi,Utmpr,Utmpi,V,s,ah2,BC,N);         
      vec_add_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0f,N);
      vec_add_F(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N);   
      }/*End of j chunk_size*/
    }else if (method==2)
    {
      for (j = 0; j<chunk_size; j++)
      {   
      compute_F_2SHOC_F(ktotr, ktoti,Uoldr,Uoldi,dx2r,dx2i,V,s,a,l_a,a76,a112,lh2,BC,N);
      vec_add_F(Utmpr,Utmpi,Uoldr,Uoldi,ktotr,ktoti,k2,N);          
      compute_F_2SHOC_F(ktmpr, ktmpi,Utmpr,Utmpi,dx2r,dx2i,V,s,a,l_a,a76,a112,lh2,BC,N);
      vec_add_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N);
      vec_add_F(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k2,N); 
      compute_F_2SHOC_F(ktmpr, ktmpi,Utmpr,Utmpi,dx2r,dx2i,V,s,a,l_a,a76,a112,lh2,BC,N);    
      vec_add_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,2.0f,N);
      vec_add_F(Utmpr,Utmpi,Uoldr,Uoldi,ktmpr,ktmpi,k,N);       
      compute_F_2SHOC_F(ktmpr, ktmpi,Utmpr,Utmpi,dx2r,dx2i,V,s,a,l_a,a76,a112,lh2,BC,N);          
      vec_add_F(ktotr,ktoti,ktotr,ktoti,ktmpr,ktmpi,1.0f,N);
      vec_add_F(Uoldr,Uoldi,Uoldr,Uoldi,ktotr,ktoti,k6,N);   
      }/*End of j chunk_size*/
	}
	
    for(j=0;j<N;j++){
      Unewr[j] = Uoldr[j];
      Unewi[j] = Uoldi[j];
    }

    /*Free up memory:*/
    free(Utmpr); free(Utmpi);
    free(ktmpr); free(ktmpi);
    free(ktotr); free(ktoti); 
	if (method==2){free(dx2r);free(dx2i);}
}








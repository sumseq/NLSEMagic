/*rmc_vorticity3D.c
Ron Caplan
Computational Science Research Center
San diego State University

A MEX function for the computation of 3D vorticity of a complex wavefunction given by:
[wx,wy,wz]=(i/2)(U*grad(U)' - U'*grad(U))/(U*conj(U)+veps)
where veps is a small value used to avoid singularities when U=0.

To use:
1) Put this file into the folder you want to use it in.
2) Compile code by typing "mex rmc_vorticity3D.c" in MATLAB. (This only needs to be done once)
3) Use as follows:
[w_x,w_y,w_z,W] = rmc_vorticity3D(U,1/(h*2),veps);
where h is the spatial step size in all directions, and veps<<1.
The w_i's are the vorticity in each direction, while W is the 2-norm of the vorticity.

Any problems, bugs, or questions can be submitted to sumseq@gmail.com
*/

#include "mex.h"
#include "math.h"

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{

    mwSize L,N,M,dims;
    mwSize *dim_array;
    int    i, j, k, itmp;
    double lh2;
    double veps, l_U2tmp;
    double *Ur,  *Ui;
    double *Gxr, *Gxi;
    double *Gyr, *Gyi;
    double *Gzr, *Gzi;
    double *W;
    double *wx,*wy,*wz;
    double *Vx,*Vy,*Vz;
    double *Vxy;
    double *Vyx;
    double *Vzx;
    double *Vxz;
    double *Vyz;
    double *Vzy;

        
    /* Find the dimensions of the vector */
    dims      = mxGetNumberOfDimensions(prhs[0]);
    dim_array = mxGetDimensions(prhs[0]);
    M = dim_array[0];
    N = dim_array[1];
    L = dim_array[2];
    
    /* Retrieve the input data */
    Ur = mxGetPr(prhs[0]);
    /*If init condition real, need to create imag aray*/
    if(mxIsComplex(prhs[0])){
        Ui = mxGetPi(prhs[0]);
    }
    else{       
        Ui = (double *)malloc(sizeof(double)*L*N*M);
        for(i=0;i<L*N*M;i++){
            Ui[i] = 0.0;
        }
    }
      
    /*Get 2Xstep sizes*/
    lh2  = (double)mxGetScalar(prhs[1]);   
    veps = (double)mxGetScalar(prhs[2]);  
    
   
    Gxr  = (double *)malloc(sizeof(double)*L*N*M);
    Gxi  = (double *)malloc(sizeof(double)*L*N*M);
    Gyr  = (double *)malloc(sizeof(double)*L*N*M);
    Gyi  = (double *)malloc(sizeof(double)*L*N*M);
    Gzr  = (double *)malloc(sizeof(double)*L*N*M);
    Gzi  = (double *)malloc(sizeof(double)*L*N*M);

    Vx   = (double *)malloc(sizeof(double)*L*N*M);
    Vy   = (double *)malloc(sizeof(double)*L*N*M);
    Vz   = (double *)malloc(sizeof(double)*L*N*M);

    Vyx  = (double *)malloc(sizeof(double)*L*N*M);
    Vxy  = (double *)malloc(sizeof(double)*L*N*M);
    Vxz  = (double *)malloc(sizeof(double)*L*N*M);
    Vzx  = (double *)malloc(sizeof(double)*L*N*M);
    Vzy  = (double *)malloc(sizeof(double)*L*N*M);
    Vyz  = (double *)malloc(sizeof(double)*L*N*M);


    /* Create an mxArray for the output data */
    plhs[0] = mxCreateNumericArray(dims, dim_array, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(dims, dim_array, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericArray(dims, dim_array, mxDOUBLE_CLASS, mxREAL);
    plhs[3] = mxCreateNumericArray(dims, dim_array, mxDOUBLE_CLASS, mxREAL);
    
    wx = mxGetPr(plhs[0]);
    wy = mxGetPr(plhs[1]);
    wz = mxGetPr(plhs[2]);
    W  = mxGetPr(plhs[3]);

    
    /*Calculate interier gradients*/
    for(i=1;i<L-1;i++){
        for(j=1;j<N-1;j++){
            for(k=1;k<M-1;k++){
              Gxr[N*M*i + M*j + k] = lh2*(Ur[N*M*(i+1) + M*j + k] - Ur[N*M*(i-1) + M*j + k]);
              Gxi[N*M*i + M*j + k] = lh2*(Ui[N*M*(i+1) + M*j + k] - Ui[N*M*(i-1) + M*j + k]);
              Gyr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*(j+1) + k] - Ur[N*M*i + M*(j-1) + k]);
              Gyi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*(j+1) + k] - Ui[N*M*i + M*(j-1) + k]);
              Gzr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*j + (k+1)] - Ur[N*M*i + M*j + (k-1)]);
              Gzi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*j + (k+1)] - Ui[N*M*i + M*j + (k-1)]);   
            }
        }
    }  
    
    /*Calculate boundary conditions with 1-sided diff:*/
    
    /*X-face:*/
    i = 0;
    for(j=1;j<N-1;j++){
         for(k=1;k<M-1;k++){
           Gxr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*(i+1) + M*j + k] - Ur[N*M*i + M*j + k]);
           Gxi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*(i+1) + M*j + k] - Ui[N*M*i + M*j + k]);  
           Gyr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*(j+1) + k] - Ur[N*M*i + M*(j-1) + k]);
           Gyi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*(j+1) + k] - Ui[N*M*i + M*(j-1) + k]);
           Gzr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*j + (k+1)] - Ur[N*M*i + M*j + (k-1)]);
           Gzi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*j + (k+1)] - Ui[N*M*i + M*j + (k-1)]); 
       }
    }    
    i = L-1;
    for(j=1;j<N-1;j++){
         for(k=1;k<M-1;k++){
           Gxr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*i + M*j + k] - Ur[N*M*(i-1) + M*j + k]);
           Gxi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*i + M*j + k] - Ui[N*M*(i-1) + M*j + k]);
           Gyr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*(j+1) + k] - Ur[N*M*i + M*(j-1) + k]);
           Gyi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*(j+1) + k] - Ui[N*M*i + M*(j-1) + k]);
           Gzr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*j + (k+1)] - Ur[N*M*i + M*j + (k-1)]);
           Gzi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*j + (k+1)] - Ui[N*M*i + M*j + (k-1)]); 
       }
    }    
    /*Y-face:*/
    j = 0;
    for(i=1;i<L-1;i++){
       for(k=1;k<M-1;k++){
           Gyr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*i + M*(j+1) + k] - Ur[N*M*i + M*j + k]);
           Gyi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*i + M*(j+1) + k] - Ui[N*M*i + M*j + k]);
           Gxr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*(i+1) + M*j + k] - Ur[N*M*i + M*j + k]);
           Gxi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*(i+1) + M*j + k] - Ui[N*M*i + M*j + k]);
           Gzr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*j + (k+1)] - Ur[N*M*i + M*j + (k-1)]);
           Gzi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*j + (k+1)] - Ui[N*M*i + M*j + (k-1)]); 
       }
    }    
    j = N-1;
    for(i=1;i<L-1;i++){
       for(k=1;k<M-1;k++){
           Gyr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*i + M*j + k] - Ur[N*M*i + M*(j-1) + k]);
           Gyi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*i + M*j + k] - Ui[N*M*i + M*(j-1) + k]); 
           Gxr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*(i+1) + M*j + k] - Ur[N*M*i + M*j + k]);
           Gxi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*(i+1) + M*j + k] - Ui[N*M*i + M*j + k]);  
           Gzr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*j + (k+1)] - Ur[N*M*i + M*j + (k-1)]);
           Gzi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*j + (k+1)] - Ui[N*M*i + M*j + (k-1)]); 
       }
    }   
    /*Z-face:*/
    k = 0;
    for(i=1;i<L-1;i++){
      for(j=1;j<N-1;j++){
           Gzr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*i + M*j + k+1] - Ur[N*M*i + M*j + k]);
           Gzi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*i + M*j + k+1] - Ui[N*M*i + M*j + k]);
           Gxr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*(i+1) + M*j + k] - Ur[N*M*i + M*j + k]);
           Gxi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*(i+1) + M*j + k] - Ui[N*M*i + M*j + k]);
           Gyr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*(j+1) + k] - Ur[N*M*i + M*(j-1) + k]);
           Gyi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*(j+1) + k] - Ui[N*M*i + M*(j-1) + k]);
       }
    }    
    k = M-1;
    for(i=1;i<L-1;i++){
       for(j=1;j<N-1;j++){
           Gzr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*i + M*j + k] - Ur[N*M*i + M*j + k-1]);
           Gzi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*i + M*j + k] - Ui[N*M*i + M*j + k-1]);
           Gxr[N*M*i + M*j + k] = (lh2*2)*(Ur[N*M*(i+1) + M*j + k] - Ur[N*M*i + M*j + k]);
           Gxi[N*M*i + M*j + k] = (lh2*2)*(Ui[N*M*(i+1) + M*j + k] - Ui[N*M*i + M*j + k]);  
           Gyr[N*M*i + M*j + k] = lh2*(Ur[N*M*i + M*(j+1) + k] - Ur[N*M*i + M*(j-1) + k]);
           Gyi[N*M*i + M*j + k] = lh2*(Ui[N*M*i + M*(j+1) + k] - Ui[N*M*i + M*(j-1) + k]);
       }
    }   
    
    /*Complete!*/
    

    /*Now compute fluid velocity*/
   for(i=0;i<L;i++){
        for(j=0;j<N;j++){
            for(k=0;k<M;k++){

                /*Store current index*/
                itmp = N*M*i+M*j+k;
                /*Compute mod-squared once for each point*/
                l_U2tmp = 1/(2*Ur[itmp]*Ur[itmp] + 2*Ui[itmp]*Ui[itmp] + veps);

                /*Compute velocity*/
                Vx[itmp] = (Ur[itmp]*Gxi[itmp] - Ui[itmp]*Gxr[itmp])*l_U2tmp;
                Vy[itmp] = (Ur[itmp]*Gyi[itmp] - Ui[itmp]*Gyr[itmp])*l_U2tmp;
                Vz[itmp] = (Ur[itmp]*Gzi[itmp] - Ui[itmp]*Gzr[itmp])*l_U2tmp;

            }
        }
   }

   /*Calculate derivatives of velocity and use them to compute wx,wy,wz,W*/
    for(i=1;i<L-1;i++){
        for(j=1;j<N-1;j++){
            for(k=1;k<M-1;k++){   
              /*Store current index*/
              itmp = N*M*i+M*j+k;     
              Vzx[itmp] = lh2*(Vz[N*M*(i+1) + M*j + k] - Vz[N*M*(i-1) + M*j + k]);              
              Vyx[itmp] = lh2*(Vy[N*M*(i+1) + M*j + k] - Vy[N*M*(i-1) + M*j + k]); 
              Vzy[itmp] = lh2*(Vz[N*M*i + M*(j+1) + k] - Vz[N*M*i + M*(j-1) + k]);
              Vxy[itmp] = lh2*(Vx[N*M*i + M*(j+1) + k] - Vx[N*M*i + M*(j-1) + k]);
              Vyz[itmp] = lh2*(Vy[N*M*i + M*j + (k+1)] - Vy[N*M*i + M*j + (k-1)]);              
              Vxz[itmp] = lh2*(Vx[N*M*i + M*j + (k+1)] - Vx[N*M*i + M*j + (k-1)]);
              wx[itmp]  = Vzy[itmp] - Vyz[itmp];
              wy[itmp]  = Vxz[itmp] - Vzx[itmp];
              wz[itmp]  = Vyx[itmp] - Vxy[itmp];     
              W[itmp]   = sqrt(wx[itmp]*wx[itmp] + wy[itmp]*wy[itmp] + wz[itmp]*wz[itmp]);
            }
        }
    }  

    /*Boundary*/
     
    /*X-face:*/
    i = 0;
    for(j=1;j<N-1;j++){
         for(k=1;k<M-1;k++){
             itmp = N*M*i+M*j+k;     
           Vzx[itmp] = (lh2*2)*(Vz[N*M*(i+1) + M*j + k] - Vz[N*M*i + M*j + k]);
           Vyx[itmp] = (lh2*2)*(Vy[N*M*(i+1) + M*j + k] - Vy[N*M*i + M*j + k]);             
           Vzy[itmp] = lh2*(Vz[N*M*i + M*(j+1) + k] - Vz[N*M*i + M*(j-1) + k]);
           Vxy[itmp] = lh2*(Vx[N*M*i + M*(j+1) + k] - Vx[N*M*i + M*(j-1) + k]);
           Vyz[itmp] = lh2*(Vy[N*M*i + M*j + (k+1)] - Vy[N*M*i + M*j + (k-1)]);              
           Vxz[itmp] = lh2*(Vx[N*M*i + M*j + (k+1)] - Vx[N*M*i + M*j + (k-1)]);
           wx[itmp]  = Vzy[itmp] - Vyz[itmp];
           wy[itmp]  = Vxz[itmp] - Vzx[itmp];
           wz[itmp]  = Vyx[itmp] - Vxy[itmp];     
           W[itmp]   = sqrt(wx[itmp]*wx[itmp] + wy[itmp]*wy[itmp] + wz[itmp]*wz[itmp]);
       }
    }    
    i = L-1;
    for(j=1;j<N-1;j++){
         for(k=1;k<M-1;k++){
             itmp = N*M*i+M*j+k;     
           Vzx[N*M*i + M*j + k] = (lh2*2)*(Vz[N*M*i + M*j + k] - Vz[N*M*(i-1) + M*j + k]);
           Vyx[N*M*i + M*j + k] = (lh2*2)*(Vy[N*M*i + M*j + k] - Vy[N*M*(i-1) + M*j + k]);           
           Vzy[itmp] = lh2*(Vz[N*M*i + M*(j+1) + k] - Vz[N*M*i + M*(j-1) + k]);
           Vxy[itmp] = lh2*(Vx[N*M*i + M*(j+1) + k] - Vx[N*M*i + M*(j-1) + k]);
           Vyz[itmp] = lh2*(Vy[N*M*i + M*j + (k+1)] - Vy[N*M*i + M*j + (k-1)]);              
           Vxz[itmp] = lh2*(Vx[N*M*i + M*j + (k+1)] - Vx[N*M*i + M*j + (k-1)]);
           wx[itmp]  = Vzy[itmp] - Vyz[itmp];
           wy[itmp]  = Vxz[itmp] - Vzx[itmp];
           wz[itmp]  = Vyx[itmp] - Vxy[itmp];     
           W[itmp]   = sqrt(wx[itmp]*wx[itmp] + wy[itmp]*wy[itmp] + wz[itmp]*wz[itmp]);
       }
    }    
    /*Y-face:*/
    j = 0;
    for(i=1;i<L-1;i++){
       for(k=1;k<M-1;k++){
           itmp = N*M*i+M*j+k;     
           Vzy[itmp] = (lh2*2)*(Vz[N*M*i + M*(j+1) + k] - Vz[N*M*i + M*j + k]);
           Vxy[itmp] = (lh2*2)*(Vx[N*M*i + M*(j+1) + k] - Vx[N*M*i + M*j + k]);
           Vzx[itmp] = (lh2*2)*(Vz[N*M*(i+1) + M*j + k] - Vz[N*M*i + M*j + k]);
           Vyx[itmp] = (lh2*2)*(Vy[N*M*(i+1) + M*j + k] - Vy[N*M*i + M*j + k]);
           Vyz[itmp] = lh2*(Vy[N*M*i + M*j + (k+1)] - Vy[N*M*i + M*j + (k-1)]);              
           Vxz[itmp] = lh2*(Vx[N*M*i + M*j + (k+1)] - Vx[N*M*i + M*j + (k-1)]);
           wx[itmp]  = Vzy[itmp] - Vyz[itmp];
           wy[itmp]  = Vxz[itmp] - Vzx[itmp];
           wz[itmp]  = Vyx[itmp] - Vxy[itmp];     
           W[itmp]   = sqrt(wx[itmp]*wx[itmp] + wy[itmp]*wy[itmp] + wz[itmp]*wz[itmp]);
       }
    }    
    j = N-1;
    for(i=1;i<L-1;i++){
       for(k=1;k<M-1;k++){
           itmp = N*M*i+M*j+k;     
           Vzy[itmp] = (lh2*2)*(Vz[N*M*i + M*j + k] - Vz[N*M*i + M*(j-1) + k]);
           Vxy[itmp] = (lh2*2)*(Vx[N*M*i + M*j + k] - Vx[N*M*i + M*(j-1) + k]); 
           Vzx[itmp] = (lh2*2)*(Vz[N*M*(i+1) + M*j + k] - Vz[N*M*i + M*j + k]);
           Vyx[itmp] = (lh2*2)*(Vy[N*M*(i+1) + M*j + k] - Vy[N*M*i + M*j + k]);  
           Vyz[itmp] = lh2*(Vy[N*M*i + M*j + (k+1)] - Vy[N*M*i + M*j + (k-1)]);              
           Vxz[itmp] = lh2*(Vx[N*M*i + M*j + (k+1)] - Vx[N*M*i + M*j + (k-1)]);
           wx[itmp]  = Vzy[itmp] - Vyz[itmp];
           wy[itmp]  = Vxz[itmp] - Vzx[itmp];
           wz[itmp]  = Vyx[itmp] - Vxy[itmp];     
           W[itmp]   = sqrt(wx[itmp]*wx[itmp] + wy[itmp]*wy[itmp] + wz[itmp]*wz[itmp]);
       }
    }   
    /*Z-face:*/
    k = 0;
    for(i=1;i<L-1;i++){
      for(j=1;j<N-1;j++){
           itmp = N*M*i+M*j+k;  
           Vyz[itmp] = (lh2*2)*(Vy[N*M*i + M*j + k+1] - Vy[N*M*i + M*j + k]);
           Vxz[itmp] = (lh2*2)*(Vx[N*M*i + M*j + k+1] - Vx[N*M*i + M*j + k]);
           Vzx[itmp] = (lh2*2)*(Vz[N*M*(i+1) + M*j + k] - Vz[N*M*i + M*j + k]);
           Vyx[itmp] = (lh2*2)*(Vy[N*M*(i+1) + M*j + k] - Vy[N*M*i + M*j + k]);
           Vzy[itmp] = lh2*(Vz[N*M*i + M*(j+1) + k] - Vz[N*M*i + M*(j-1) + k]);
           Vxy[itmp] = lh2*(Vx[N*M*i + M*(j+1) + k] - Vx[N*M*i + M*(j-1) + k]);
           wx[itmp]  = Vzy[itmp] - Vyz[itmp];
           wy[itmp]  = Vxz[itmp] - Vzx[itmp];
           wz[itmp]  = Vyx[itmp] - Vxy[itmp];     
           W[itmp]   = sqrt(wx[itmp]*wx[itmp] + wy[itmp]*wy[itmp] + wz[itmp]*wz[itmp]);
       }
    }    
    k = M-1;
    for(i=1;i<L-1;i++){
       for(j=1;j<N-1;j++){
           itmp = N*M*i+M*j+k;     
           Vyz[itmp] = (lh2*2)*(Vy[N*M*i + M*j + k] - Vy[N*M*i + M*j + k-1]);
           Vxz[itmp] = (lh2*2)*(Vx[N*M*i + M*j + k] - Vx[N*M*i + M*j + k-1]);
           Vzx[itmp] = (lh2*2)*(Vz[N*M*(i+1) + M*j + k] - Vz[N*M*i + M*j + k]);
           Vyx[itmp] = (lh2*2)*(Vy[N*M*(i+1) + M*j + k] - Vy[N*M*i + M*j + k]);  
           Vzy[itmp] = lh2*(Vz[N*M*i + M*(j+1) + k] - Vz[N*M*i + M*(j-1) + k]);
           Vxy[itmp] = lh2*(Vx[N*M*i + M*(j+1) + k] - Vx[N*M*i + M*(j-1) + k]);
           wx[itmp]  = Vzy[itmp] - Vyz[itmp];
           wy[itmp]  = Vxz[itmp] - Vzx[itmp];
           wz[itmp]  = Vyx[itmp] - Vxy[itmp];     
           W[itmp]   = sqrt(wx[itmp]*wx[itmp] + wy[itmp]*wy[itmp] + wz[itmp]*wz[itmp]);
       }
    }   
    
    /*Complete!*/

   /*Free up memory*/
   free(Gxr); free(Gxi); 
   free(Gyr); free(Gyi);
   free(Gzr); free(Gzi);

   free(Vx);
   free(Vy);
   free(Vz);
   free(Vxz);
   free(Vyz);
   free(Vzx);
   free(Vxy);
   free(Vyx);
   free(Vzy);
    
   if(!mxIsComplex(prhs[0])){
       free(Ui);
   }

}  

#include "mex.h"
#include "NLSEmagic.h"

/*Main mex function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int     i,j, N, chunk_size, BC, precision, method;
    double  h2, a, s, k;
    double *Uoldr, *Uoldi, *V, *Unewr, *Unewi;
    float  *fUoldr,*fUoldi,*fV,*fUnewr,*fUnewi;

    /* Find the dimensions of the vector */
    N = mxGetN(prhs[0]);
    if(N==1){
      N = mxGetM(prhs[0]);
    }

    /* Retrieve the input data */
    Uoldr = mxGetPr(prhs[0]);

    /*Check if initial condition fully real or not*/
    if(mxIsComplex(prhs[0])){
        Uoldi = mxGetPi(prhs[0]);
    }
    else{  /*Need to create imaginary part vector*/
        Uoldi = (double*)malloc(N*sizeof(double));
        for(j=0;j<N;j++){
           Uoldi[j] = 0.0;
        }
    }

    /*Get the rest of the input variables*/
    V           = mxGetPr(prhs[1]);
    s           = (double)mxGetScalar(prhs[2]);
    a           = (double)mxGetScalar(prhs[3]);
    h2          = (double)mxGetScalar(prhs[4]);
    BC          =    (int)mxGetScalar(prhs[5]);
    chunk_size  =    (int)mxGetScalar(prhs[6]);
    k           = (double)mxGetScalar(prhs[7]);
    precision   =    (int)mxGetScalar(prhs[8]);
    method      =    (int)mxGetScalar(prhs[9]);

    /*Create result vector*/
    plhs[0] = mxCreateDoubleMatrix(N,1,mxCOMPLEX);

    /*Get pointers to result vectors*/
    Unewr = mxGetPr(plhs[0]);
    Unewi = mxGetPi(plhs[0]);

    if(precision==1){
      /*Create float array pointers*/
      fUoldr  = (float *) malloc(sizeof(float)*N);
      fUoldi  = (float *) malloc(sizeof(float)*N);
      fUnewr  = (float *) malloc(sizeof(float)*N);
      fUnewi  = (float *) malloc(sizeof(float)*N);
      fV      = (float *) malloc(sizeof(float)*N);

      /*Convert initial condition to float type:*/
      for(i=0;i<N;i++){
        fUoldr[i] = (float) Uoldr[i];
        fUoldi[i] = (float) Uoldi[i];
            fV[i] = (float) V[i];
      }

      /*Call integrator*/
      NLSE1D_TAKE_STEPS_CUDA_F(fUoldr,fUoldi,fV,fUnewr,fUnewi,(float)s,(float)a,
                               (float)h2,BC,chunk_size,(float)k,N,method);

      /*Convert result to double*/
      for(i=0;i<N;i++){
        Unewr[i] = (double) fUnewr[i];
        Unewi[i] = (double) fUnewi[i];
      }
       /*Free up float arrays*/
       free(fUoldr);
       free(fUoldi);
       free(fV);
       free(fUnewr);
       free(fUnewi);
    }
    else if(precision==2){
      /*Call integrator*/
      NLSE1D_TAKE_STEPS_CUDA(Uoldr,Uoldi,V,Unewr,Unewi,s,a,h2,BC,chunk_size,k,N,
                             method);
    }
	
    if(!mxIsComplex(prhs[0])){
      free(Uoldi);
    }
}



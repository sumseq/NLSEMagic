#include "mex.h"
#include "NLSEmagic.h"

/*Main mex function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int     i, N, M, chunk_size, BC, precision, method;
    double  h2, a, s, k;
    double *vUnewr, *vUnewi, *vUoldr, *vUoldi, *vV;
	float  *fUoldr, *fUoldi, *fV, *fUnewr, *fUnewi;

    /* Find the dimensions of the array */
    N = mxGetN(prhs[0]);
    M = mxGetM(prhs[0]);

    /* Retrieve the input data */
    vUoldr = mxGetPr(prhs[0]);

    /*Check if initial condition fully real or not*/
    if(mxIsComplex(prhs[0])){
        vUoldi = mxGetPi(prhs[0]);
    }
    else{  /*Need to create imaginary part vector*/
        vUoldi = (double*)malloc(N*M*sizeof(double));
        for(i=0;i<N*M;i++){
           vUoldi[i] = 0.0;
        }
    }

    /*Get the rest of the input variables*/
    vV          = mxGetPr(prhs[1]);
    s           = (double)mxGetScalar(prhs[2]);
    a           = (double)mxGetScalar(prhs[3]);
    h2          = (double)mxGetScalar(prhs[4]);
    BC          =    (int)mxGetScalar(prhs[5]);
    chunk_size  =    (int)mxGetScalar(prhs[6]);
    k           = (double)mxGetScalar(prhs[7]);
    precision   =    (int)mxGetScalar(prhs[8]);
    method      =    (int)mxGetScalar(prhs[9]);

    /* Create an mxArray for the output data */
    plhs[0] = mxCreateDoubleMatrix(M,N,mxCOMPLEX);

    /*Get pointers to result vectors*/
    vUnewr = mxGetPr(plhs[0]);
    vUnewi = mxGetPi(plhs[0]);	

    if(precision==1){
      /*Create float array pointers*/
      fUoldr  = (float *) malloc(sizeof(float)*N*M);
      fUoldi  = (float *) malloc(sizeof(float)*N*M);
      fV      = (float *) malloc(sizeof(float)*N*M);

      /*Convert initial condition to float type:*/
      for(i=0;i<N*M;i++){
        fUoldr[i] = (float) vUoldr[i];
        fUoldi[i] = (float) vUoldi[i];
            fV[i] = (float) vV[i];
      }

      /*Call integrator*/
      NLSE2D_TAKE_STEPS_CUDA_F(fUoldr,fUoldi,fV,vUnewr,vUnewi,(float)s,(float)a,(float)h2,BC,
                               chunk_size,(float)k,N,M,method);

      /*Free up float arrays*/
      free(fUoldr);
      free(fUoldi);
      free(fV);
      }
    else if(precision==2){
        /*Call integrator*/
        NLSE2D_TAKE_STEPS_CUDA(vUoldr,vUoldi,vV,vUnewr,vUnewi,s,a,h2,BC,chunk_size,k,N,M,method);
    }
	
    if(!mxIsComplex(prhs[0])){
      free(vUoldi);
    }
}

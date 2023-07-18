extern void NLSE1D_TAKE_STEPS_CUDA(double*, double*, double*, double*, double*,  
	                           double, double, double, int, int, double, int, int);

extern void NLSE1D_TAKE_STEPS_CUDA_F(float*, float*, float*, float*, float*,  
	                             float, float, float, int, int, float, int, int);

extern void NLSE1D_TAKE_STEPS(double*, double*, double*, double*, double*,  
	                      double, double, double, int, int, double, int, int);

extern void NLSE1D_TAKE_STEPS_F(float*, float*, float*, float*, float*,  
	                        float, float, float, int, int, float, int, int);
							
extern void NLSE2D_TAKE_STEPS_CUDA(double*, double*, double*, double*, double*,  
	                           double, double, double, int, int, double, int, int, int);

extern void NLSE2D_TAKE_STEPS_CUDA_F(float*, float*, float*, float*, float*,  
	                             float, float, float, int, int, float, int, int, int);

extern void NLSE2D_TAKE_STEPS(double*, double*, double*, double*, double*,  
	                      double, double, double, int, int, double, int, int, int);

extern void NLSE2D_TAKE_STEPS_F(float*, float*, float*, double*, double*,  
	                        float, float, float, int, int, float, int, int, int);

extern void NLSE3D_TAKE_STEPS_CUDA(double*, double*, double*, double*, double*,  
	                           double, double, double, int, int, double, int, int, int, int);

extern void NLSE3D_TAKE_STEPS_CUDA_F(float*, float*, float*, float*, float*,  
	                             float, float, float, int, int, float, int, int, int, int);

extern void NLSE3D_TAKE_STEPS(double*, double*, double*, double*, double*,  
	                      double, double, double, int, int, double, int, int, int, int);

extern void NLSE3D_TAKE_STEPS_F(float*, float*, float*, double*, double*,  
	                        float, float, float, int, int, float, int, int, int, int);
# Define installation location for CUDA and compilation flags compatible
# with the CUDA include files.
CUDAHOME    = /usr/local/cuda
INCLUDEDIR  = -I$(CUDAHOME)/include
INCLUDELIB  = -L/usr/local/MATLAB/R2012a_Student/bin/glnx86 -L$(CUDAHOME)/lib -lcuda -lcufft -lcudart -Wl,-rpath,$(CUDAHOME)/lib
CFLAGS      = -m32 -fPIC -D_GNU_SOURCE -pthread -fexceptions
COPTIMFLAGS = -O3 -funroll-loops -msse2

# Define installation location for MATLAB.
export MATLAB = /usr/local/MATLAB/R2012a_Student
MEX           = $(MATLAB)/bin/mex
MEXEXT        = .mexglx #.$(shell $(MATLAB)/bin/mexext)

# nvmex is a modified mex script that knows how to handle CUDA .cu files.
NVMEX = ./nvmex

# List the mex files to be built.  The .mex extension will be replaced with the
# appropriate extension for this installation of MATLAB, e.g. .mexglx or
# .mexa64.
MEXFILES = NLSE2D_TAKE_STEPS_CD_CUDA_F.mex NLSE2D_TAKE_STEPS_CD_CUDA_D.mex NLSE2D_TAKE_STEPS_2SHOC_CUDA_F.mex NLSE2D_TAKE_STEPS_2SHOC_CUDA_D.mex getCudaInfo.mex

all: $(MEXFILES:.mex=$(MEXEXT))

clean:
	rm -f $(MEXFILES:.mex=$(MEXEXT))

.SUFFIXES: .cu .cu_o .mexglx .mexa64

.c.mexglx:
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
	$(INCLUDEDIR) $(INCLUDELIB)

.cu.mexglx:
	$(NVMEX) -glnx86 -f ./nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB)

.c.mexa64:
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
	$(INCLUDEDIR) $(INCLUDELIB)

.cu.mexa64:
	$(NVMEX) -f nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB)

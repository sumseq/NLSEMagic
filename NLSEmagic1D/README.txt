NLSEmagic1D Readme
------------------

Basic installation instructions (for Windows and Linux):
1) Extract all files in this zip file into a directory of your choice.
2) Run MATLAB and change the directory to where you unzipped the files.
3) Type "install_NLSEmagic1D"
4) Now you can run NLSEmagic1D.

Alter the m file NLSEmagic1D.m to run your own simulations.

For the codes to compile, you must have installed the required libraries.

For examples on how to plot, save movies/images, run timings, etc. 
see the full research script code package.


Some notes:

GPU_ARCH=sm_30
CPU_ARCH=32
CUDA_HOME_DIR=/usr/local/cuda
VISUAL_CPP_HOME_DIR='C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin'

nvcc -c NLSEmagic1D_GPU.cu -arch=sm_30 -O3 -m32
mex NLSEmagic1D_GPU.o NLSEmagic1D_Take_Steps_GPU.c -L/usr/local/cuda/lib -lcudart

gcc -c NLSEmagic1D.c -O3 -m32
mex NLSEmagic1D_CPU.o NLSEmagic1D_Take_Steps_CPU.c

nvcc -m32 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin" -arch=sm_21 -O3 -c NLSEmagic1D_GPU.cu 
mex NLSEmagic1D_Take_Steps_GPU.c NLSEmagic1D_GPU.obj -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\lib\Win32" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\include" -lcudart


"C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\vcvarsall.bat"
"C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\cl.exe" -c NLSEmagic1D_CPU.c /O2 /TP
mex NLSEmagic1D_Take_Steps_CPU.c NLSEmagic1D_CPU.obj

-------------------------------------------------------------------------
Disclaimer - This code is given as "as is" and is not guaranteed at all 
             and I take no liability for any damage it may cause. Feel 
			 free to distribute the code as long as you keep the authors 
			 name in it.  If you use these codes, the author would appreciate 
			 your acknowledging having done so in any reports, publications, etc. 
			 resulting from their use.  

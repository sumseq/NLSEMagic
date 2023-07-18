%install_NLSEmagic1D.m
%Script to Compile 1D NLSE Integrators
%2014 Ronald M Caplan
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

opsys=computer;  

cuda_avail=input('Do you have an NVIDIA CUDA GPU card with CC 2.0 and up? (y/n): ','s');

%Cleanup from any previous compilations:
if(strcmp(opsys,'GLNX86') || strcmp(opsys,'GLNXA64'))
   system('rm *.o *.obj 2>/dev/null');   
end
if(strcmp(opsys,'PCWIN') || strcmp(opsys,'PCWIN64'))
   system('del *.o *.obj');   
end

fprintf('Compiling CPU integrators...');
if(strcmp(opsys,'PCWIN') || strcmp(opsys,'PCWIN64'))  
  mex -c NLSEmagic1D_CPU.c COMPFLAGS="$COMPFLAGS /TP"
  %system('"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"')
  %system('"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\cl.exe" -c NLSEmagic1D_CPU.c /O2 /TP')
else
  %system('gcc -c NLSEmagic1D.c -O3 -DNDEBUG -m32')
  mex -c NLSEmagic1D_CPU.c CC='g++' COMPFLAGS='$COMPFLAGS -O3 -DNDEBUG'
end
fprintf('Done!\n');

fprintf('Creating MEX interface to CPU integrators...');
if(strcmp(opsys,'PCWIN') || strcmp(opsys,'PCWIN64')) 
  mex NLSEmagic1D_Take_Steps_CPU.c NLSEmagic1D_CPU.obj
else
  mex NLSEmagic1D_Take_Steps_CPU.c NLSEmagic1D_CPU.o   
end
fprintf('Done!\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(cuda_avail=='y')

fprintf('Compiling GPU integrators...');
if(strcmp(opsys,'GLNX86'))
    system('nvcc -m32 -Xcompiler -fPIC -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=sm_21 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_32,code=sm_32 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -O3 -c NLSEmagic1D_GPU.cu'); 
elseif(strcmp(opsys,'GLNXA64'))
    system('nvcc -m64 -Xcompiler -fPIC -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=sm_21 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_32,code=sm_32 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -O3 -c NLSEmagic1D_GPU.cu'); 
elseif(strcmp(opsys,'PCWIN'))
    system('nvcc -m32 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=sm_21 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_32,code=sm_32 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -O3 -c NLSEmagic1D_GPU.cu'); 
elseif(strcmp(opsys,'PCWIN64'))
    system('nvcc -m64 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin" -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=sm_21 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_32,code=sm_32 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -O3 -c NLSEmagic1D_GPU.cu'); 
end
fprintf('Done!\n');

fprintf('Creating MEX interface to GPU integrators...');
if(strcmp(opsys,'PCWIN')) 
  mex NLSEmagic1D_Take_Steps_GPU.c NLSEmagic1D_GPU.obj -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\Win32" -lcudart
elseif(strcmp(opsys,'PCWIN64'))
  mex NLSEmagic1D_Take_Steps_GPU.c NLSEmagic1D_GPU.obj -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\Win64" -lcudart
elseif(strcmp(opsys,'GLNX86'))
  mex NLSEmagic1D_Take_Steps_GPU.c NLSEmagic1D_GPU.o -L/usr/local/cuda/lib -lcudart
elseif(strcmp(opsys,'GLNXA64'))
  mex NLSEmagic1D_Take_Steps_GPU.c NLSEmagic1D_GPU.o -L/usr/local/cuda/lib64 -lcudart
end
fprintf('Done!\n');

%Cleanup:
if(strcmp(opsys,'GLNX86') || strcmp(opsys,'GLNXA64'))
   system('rm *.o *.obj 2>/dev/null');   
end
if(strcmp(opsys,'PCWIN') || strcmp(opsys,'PCWIN64'))
   system('del *.o *.obj');   
end

end

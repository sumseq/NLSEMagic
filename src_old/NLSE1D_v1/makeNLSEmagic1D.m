%makeNLSEmagic1D  
%Script to Compile 1D NLSE Integrators
%�2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

close all;
clear all;
clear mex;

disp('Compiling mex files:');
fprintf('Compiling NLSE1D_TAKE_STEPS_CD.c.............');
mex NLSE1D_TAKE_STEPS_CD.c
fprintf('Done!\n');
fprintf('Compiling NLSE1D_TAKE_STEPS_CD_F.c...........');
mex NLSE1D_TAKE_STEPS_CD_F.c
fprintf('Done!\n');
fprintf('Compiling NLSE1D_TAKE_STEPS_2SHOC.c..........');
mex NLSE1D_TAKE_STEPS_2SHOC.c
fprintf('Done!\n');
fprintf('Compiling NLSE1D_TAKE_STEPS_2SHOC_F.c........');
mex NLSE1D_TAKE_STEPS_2SHOC_F.c
fprintf('Done!\n');
disp('Mex compilation completed!');

str = computer;
if( strcmp(str, 'PCWIN') || strcmp(str, 'PCWIN64'))
    if(exist([matlabroot '/bin/nvmex.pl'],'file')~=0)
       disp('Compiling CUDA mex files:');
       if(strcmp(str, 'PCWIN64'))
             nvmex -f nvmexopts64.bat getCudaInfo.cu -IC:\cuda\include -LC:\cuda\lib\x64 -lcudart -lcuda
        	 nvmex -f nvmexopts64.bat NLSE1D_TAKE_STEPS_CD_CUDA_F.cu -IC:\cuda\include -LC:\cuda\lib\x64 -lcudart
       		 nvmex -f nvmexopts64.bat NLSE1D_TAKE_STEPS_CD_CUDA_D.cu -IC:\cuda\include -LC:\cuda\lib\x64 -lcudart
	         nvmex -f nvmexopts64.bat NLSE1D_TAKE_STEPS_2SHOC_CUDA_F.cu -IC:\cuda\include -LC:\cuda\lib\x64 -lcudart
             nvmex -f nvmexopts64.bat NLSE1D_TAKE_STEPS_2SHOC_CUDA_D.cu -IC:\cuda\include -LC:\cuda\lib\x64 -lcudart
       elseif(strcmp(str, 'PCWIN'))
             nvmex -f nvmexopts.bat getCudaInfo.cu -IC:\cuda\include -LC:\cuda\lib\Win32 -lcudart -lcuda
        	 nvmex -f nvmexopts.bat NLSE1D_TAKE_STEPS_CD_CUDA_F.cu -IC:\cuda\include -LC:\cuda\lib\Win32 -lcudart
       		 nvmex -f nvmexopts.bat NLSE1D_TAKE_STEPS_CD_CUDA_D.cu -IC:\cuda\include -LC:\cuda\lib\Win32 -lcudart
	         nvmex -f nvmexopts.bat NLSE1D_TAKE_STEPS_2SHOC_CUDA_F.cu -IC:\cuda\include -LC:\cuda\lib\Win32 -lcudart
             nvmex -f nvmexopts.bat NLSE1D_TAKE_STEPS_2SHOC_CUDA_D.cu -IC:\cuda\include -LC:\cuda\lib\Win32 -lcudart             
       end
       disp('Done');
    else
       disp('No CUDA detected.');
    end
else
    disp('Compiling CUDA mex files:');
    system('make clean');
    system('make');
end

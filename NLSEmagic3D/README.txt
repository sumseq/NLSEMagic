NLSEmagic3D Readme
------------------

Basic installation instructions (for Windows and Linux):
1) Extract all files in this zip file into a directory of your choice.
2) Run MATLAB and change the directory to where you unzipped the files.
3) Type "makeNLSEmagic3D"
4) Now you can run NLSEmagic3D.

NOTE:  GPU-accelerated codes will NOT compile until after you follow the instructions 
       given in the "CUDA_MATLAB_SETUP_GUIDE.pdf" included in this zip file.  
	   Otherwise, only CPU mex codes will compile.
	   The integrator codes can be used without compilation by downloading the pre-compiled libraries
	   at www.nlsemagic.com.

Alter the m file NLSEmagic3D.m to run your own simulations.

For examples on how to plot, save movies/images, run timings, etc. 
see the full research script code package of NLSE3D at www.nlsemagic.com.

-------------------------------------------------------------------------
Disclaimer - This code is given as "as is" and is not guaranteed at all 
             and I take no liability for any damage it may cause. Feel 
			 free to distribute the code as long as you keep the authors 
			 name in it.  If you use these codes, the author would appreciate 
			 your acknowledging having done so in any reports, publications, etc. 
			 resulting from their use.  
			 Also, donations are welcome at www.nlsemagic.com.

![NLSEMagic](nlsemagiclogo.jpg)
# Nonlinear Schrödinger Equation Multidimensional Matlab-based GPU-accelerated Integrators using Compact high-order schemes
  
NLSEmagic is a package of C and MATLAB script codes which simulate the nonlinear Schrödinger equation in one, two, and three dimensions.  The code includes MEX integrators in C, as well as NVIDIA CUDA-enabled GPU-accelerated MEX files in C.  The MATLAB script files call the compiled MEX codes forming an easy-to-use highly efficient program.  The codes utilize a fourth-order (in time) Runge-Kutta scheme combined with the choice of standard second-order (in space) finite differencing, or a compact  two-step fourth-order (in space) finite differencing.  
The code was developed as part of my Ph.D. dissertation, and includes two versions.  One is a streamlined easy-to-follow script code which is meant as an example of how to use the MEX codes, while the other version is a full-research code which can reproduce my research results.  
  
NLSEmagic is freely distributed for use and modification.  However, we ask that you cite the following paper in publications and include an acknowledgment of authorship in any code derived from NLSEMagic:
  
NLSEmagic: Nonlinear Schrödinger Equation Multidimensional Matlab-based GPU-accelerated Integrators using Compact High-order Schemes
R.M. Caplan. Computer Physics Communications. 184,4 (2013) 1250-1271.  

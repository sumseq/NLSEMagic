@echo off
rem
rem MEX OPTS for use with nvmex (nvcc CUDA compiler) on 32-bit windows. 
rem General parameters
rem ********************************************************************

set MATLAB=%MATLAB%

rem *****These two lines are all that should need to be tweaked (and target GPU below):
rem *****(Make sure you have sintalled the MS SDK and VS express 8)

set VSINSTALLDIR=C:\Program Files\Microsoft Visual Studio 9.0
set SDKDIR=C:\Program Files\Microsoft SDKs\Windows\v6.0A

set VCINSTALLDIR=%VSINSTALLDIR%\VC
set PATH=%VCINSTALLDIR%\BIN\;%VCINSTALLDIR%\PlatformSDK\bin;%VSINSTALLDIR%\Common7\IDE;%VSINSTALLDIR%\SDK\v2.0\bin;%VSINSTALLDIR%\Common7\Tools;%VSINSTALLDIR%\Common7\Tools\bin;%VCINSTALLDIR%\VCPackages;%MATLAB_BIN%;%SDKDIR%\bin\;%PATH%

set INCLUDE=%VCINSTALLDIR%\ATLMFC\INCLUDE;%VCINSTALLDIR%\INCLUDE;%VCINSTALLDIR%\PlatformSDK\INCLUDE;%VSINSTALLDIR%\SDK\v2.0\include;%SDKDIR%\include;%INCLUDE%
set LIB=%VCINSTALLDIR%\ATLMFC\LIB;%VCINSTALLDIR%\LIB;%VCINSTALLDIR%\PlatformSDK\lib;%VSINSTALLDIR%\SDK\v2.0\lib;%MATLAB%\extern\lib\win32;%LIB%;%SDKDIR%\Lib

set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=nvcc
set COMPFLAGS=-gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20 -gencode=arch=compute_10,code=compute_10 -gencode=arch=compute_13,code=compute_13 -c -Xcompiler "/c /Zp8 /GR /W3 /EHs /D_CRT_SECURE_NO_DEPRECATE /D_SCL_SECURE_NO_DEPRECATE /D_SECURE_SCL=0 /DMATLAB_MEX_FILE /nologo /MD"
set OPTIMFLAGS=-Xcompiler "/O2 /Oy- /DNDEBUG"
set DEBUGFLAGS=-Xcompiler "/Zi /Fd"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb""
set NAME_OBJECT= 

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\microsoft
set LINKER=link
set LINKFLAGS=/dll /export:%ENTRYPOINT% /MAP /LIBPATH:"%LIBLOC%" libmx.lib libmex.lib libmat.lib /implib:%LIB_NAME%.x /MACHINE:X86 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib

set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/DEBUG /PDB:"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb"
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%%MEX_EXT%"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=rc /fo "%OUTDIR%mexversion.res"
set RC_LINKER=

set POSTLINK_CMDS=del "%OUTDIR%%MEX_NAME%.map"
set POSTLINK_CMDS1=del %LIB_NAME%.x
set POSTLINK_CMDS2=mt -outputresource:"%OUTDIR%%MEX_NAME%%MEX_EXT%";2 -manifest "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest"
set POSTLINK_CMDS3=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest" 

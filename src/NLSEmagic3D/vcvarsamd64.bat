@SET VSINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 9.0
@SET VCINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC
@SET WIN64SDKDIR=C:\Program Files\Microsoft SDKs\Windows\v7.0

@SET FrameworkDir=C:\WINDOWS\Microsoft.NET\Framework64
@SET FrameworkVersion=v3.5
@if "%VSINSTALLDIR%"=="" goto error_no_VSINSTALLDIR
@if "%VCINSTALLDIR%"=="" goto error_no_VCINSTALLDIR

@echo Setting environment for using Microsoft Visual Studio 2008 x64 tools.

@set PATH=%VCINSTALLDIR%\BIN\amd64;%FrameworkDir%\%FrameworkVersion%;%VCINSTALLDIR%\VCPackages;%VSINSTALLDIR%\Common7\IDE;%VSINSTALLDIR%\Common7\Tools;%WIN64SDKDIR%\Bin\x64;%WIN64SDKDIR%\Bin\x64\vsstools;c:\CUDA\bin;%PATH%
@set INCLUDE=%VCINSTALLDIR%\INCLUDE;%WIN64SDKDIR%\INCLUDE;%INCLUDE%
@set LIB=%VCINSTALLDIR%\LIB\amd64;%WIN64SDKDIR%\Lib\x64;%LIB%

@set LIBPATH=%FrameworkDir%\%Framework35Version%;%VCINSTALLDIR%\ATLMFC\LIB\amd64;%VCINSTALLDIR%\LIB\amd64;%LIBPATH%

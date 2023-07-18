%CQNLS_STEADY_F Stead-state 1D ODE for finding vortex radial profiles.
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function CQNLS_STEADY_F = CQNLS_STEADY_F(Uguess,par)

m   = par(1);
OM  = par(2);
a   = par(3);
s   = par(4);
h_r = par(5);
lBC = par(6);
rBC = par(7);


%Only need interier points since bounderies taken to be exact:
Uguess = squeeze(Uguess(2:end-1));

maxri = length(Uguess);
rvec  = [h_r:h_r:maxri*h_r]';
rvecF = rvec+(h_r/2);
rvecB = rvec-(h_r/2);
ri_in = [2:maxri-1]';

Ulap = zeros([maxri 1]);

%Calculate Laplacian of interier points:
Ulap(ri_in) = ((1./rvec(ri_in)).*(1/h_r).*(...
rvecF(ri_in).*(1/h_r).*(Uguess(ri_in+1) - Uguess(ri_in)) - ...
rvecB(ri_in).*(1/h_r).*(Uguess(ri_in) - Uguess(ri_in-1))));

%Boundary Condition:
%(point before 1st = lBC, after last =rBC)
%Ulap(-1) = lBC
Ulap(1) = ((1./rvec(1)).*(1/h_r).*(...
rvecF(1).*(1/h_r).*(Uguess(2) - Uguess(1)) - ...
rvecB(1).*(1/h_r).*(Uguess(1) - lBC)));

%Ulap(n+1) = rBC
Ulap(end) = ((1./rvec(maxri)).*(1/h_r).*(...
rvecF(end).*(1/h_r).*(rBC - Uguess(end)) - ...
rvecB(end).*(1/h_r).*(Uguess(end) - Uguess(end-1))));

%Send back F:
CQNLS_STEADY_F = -OM*Uguess - (a*m^2./rvec.^2).*Uguess + a*Ulap + s.*(Uguess.^3 - Uguess.^5);
CQNLS_STEADY_F = [0;CQNLS_STEADY_F;0];

return;
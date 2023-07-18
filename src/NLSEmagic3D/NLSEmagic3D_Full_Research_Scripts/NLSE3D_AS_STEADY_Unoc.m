%NLSE3D_AS_STEADY_Unoc  Steady-state 3D NLSE
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function NLSE3D_AS_STEADY_Unoc = NLSE3D_AS_STEADY_Unoc(U,par)

OM  = par(1);
a   = par(2);
s   = par(3);
h   = par(4);
M   = par(5);
N   = par(6);
c   = par(7);
rmin= par(10);
rmax= par(11);
zmin= par(12);
zmax= par(13);

rvec  = rmin:h:rmax;
zvec  = zmin:h:zmax;

[Z,R] = meshgrid(zvec,rvec);

half = floor((length(U))/2);
U    = U(1:half) + 1i*U(half + 1:end);
U    = reshape(U,M,N);

%Calculate laplacian of interier points:
%Get size of data cube:
sizeu = size(U);
%Initialize Ut and (temp) Laplacian cubes:
LAP   = zeros(sizeu);
%Make interier indicies:
rin   = [2:sizeu(1)-1];
zin   = [2:sizeu(2)-1];
%Initialize Gt and (temp) Laplacian cubes:
Ur    = zeros(sizeu);
Urr   = zeros(sizeu);
Uzz   = zeros(sizeu);

%compute first and second derivatives:
Ur(rin,:) = (U(rin+1,:) - U(rin-1,:))/(2*h);
Urr(rin,:) = (U(rin+1,:) - 2*U(rin,:) + U(rin-1,:))/(h^2);
Uzz(:,zin) = (U(:,zin+1) - 2*U(:,zin) + U(:,zin-1))/(h^2);

%Compute quasi-2D second-order CD Laplacian on axisymmetric cyllidrical coordinates:
LAP(rin,zin) =  Ur(rin,zin)./R(rin,zin) + Urr(rin,zin) + Uzz(rin,zin);
              
% %Boundary Condition:
LAP(rin,1)   = U(rin,  1).*(LAP(rin,    2)./U(rin,2)     + (s/a)*(U(rin,    2).*conj(U(rin,    2)) - U(rin,  1).*conj(U(rin,  1))));
LAP(1,zin)   = U(1,  zin).*(LAP(2,    zin)./U(2,zin)     + (s/a)*(U(2,    zin).*conj(U(2,    zin)) - U(1,  zin).*conj(U(1,  zin))));
LAP(rin,end) = U(rin,end).*(LAP(rin,end-1)./U(rin,end-1) + (s/a)*(U(rin,end-1).*conj(U(rin,end-1)) - U(rin,end).*conj(U(rin,end))));
LAP(end,zin) = U(end,zin).*(LAP(end-1,zin)./U(end-1,zin) + (s/a)*(U(end-1,zin).*conj(U(end-1,zin)) - U(end,zin).*conj(U(end,zin))));
%Corners
LAP(1,1)     = U(2,        2).*(LAP(2,        2)./U(2,        2) + (s/a)*(U(2,        2).*conj(U(2,        2)) - U(1,    1).*conj(U(1,    1))));
LAP(end,1)   = U(end-1,    2).*(LAP(end-1,    2)./U(end-1,    2) + (s/a)*(U(end-1,    2).*conj(U(end-1,    2)) - U(end,  1).*conj(U(end,  1))));
LAP(1,end)   = U(2,    end-1).*(LAP(2,    end-1)./U(2,    end-1) + (s/a)*(U(2,    end-1).*conj(U(2,    end-1)) - U(1,  end).*conj(U(1,  end))));
LAP(end,end) = U(end-1,end-1).*(LAP(end-1,end-1)./U(end-1,end-1) + (s/a)*(U(end-1,end-1).*conj(U(end-1,end-1)) - U(end,end).*conj(U(end,end))));

%Send back F:
NLSE3D_AS_STEADY_Unoc = -OM*U + a*LAP + s.*U.^2.*conj(U);
NLSE3D_AS_STEADY_Unoc = [real(NLSE3D_AS_STEADY_Unoc(:)) ; imag(NLSE3D_AS_STEADY_Unoc(:))];

return;

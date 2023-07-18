%NLSE3D_AS_STEADY_Gc Steady-state PDE of VR with z-vel of c.
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function NLSE3D_AS_STEADY_Gc = NLSE3D_AS_STEADY_Gc(G,par)

OM  = par(1);
a   = par(2);
s   = par(3);
h   = par(4);
M   = par(5);
N   = par(6);
c   = par(7);
d   = par(8);
m   = par(9);
rmin= par(10);
rmax= par(11);
zmin= par(12);
zmax= par(13);

rvec  = rmin:h:rmax;
zvec  = zmin:h:zmax;

[Z,R] = meshgrid(zvec,rvec);
P2    = (R-d).^2 + Z.^2;

%Formulate 2D G:
half = floor((length(G))/2);
G = G(1:half) + 1i*G(half + 1:end);
G = reshape(G,M,N);

%Calculate laplacian of interier points:
%Get size of data cube:
sizeg    = size(G);
%Initialize Gt and (temp) Laplacian cubes:
Gr       = zeros(sizeg);
Gz       = zeros(sizeg);
Grr      = zeros(sizeg);
Gzz      = zeros(sizeg);
%Make interier indicies:
rin      = [2:sizeg(1)-1];
zin      = [2:sizeg(2)-1];

%compute first and second derivatives:
Gr(rin,:) = (G(rin+1,:) - G(rin-1,:))/(2*h);
Gz(:,zin) = (G(:,zin+1) - G(:,zin-1))/(2*h);

Grr(rin,:) = (G(rin+1,:) - 2*G(rin,:) + G(rin-1,:))/(h^2);
Gzz(:,zin) = (G(:,zin+1) - 2*G(:,zin) + G(:,zin-1))/(h^2);

%Boundary conditions:
Gr(1,:)    = 0;% (-3*G(1,:) + 4*G(2,:) -G(3,:))/(2*h);%Gr(2,:); %due to symmetry
Grr(1,:)   = 0;%(2*G(1,:) - 5*G(2,:) +4*G(3,:) - G(4,:))/(h^2);%Grr(2,:);%(G(2,:) - 2*G(1,:) + G(2,:))/(h^2); %due to symmetry
Gr(end,:)  = (-3*G(end,:) + 4*G(end-1,:) -G(end-2,:))/(2*h);%Gr(end-1,:); %guess
Grr(end,:) = (2*G(end,:) - 5*G(end-1,:) +4*G(end-2,:) - G(end-3,:))/(h^2);%Grr(end-1,:); %guess

Gz(:,1)    = 2*Gz(:,2) - Gz(:,3);%  (-3*G(:,1)   + 4*G(:,2)     - G(:,3))    /(2*h);%Gz(:,2); %All guesses
Gzz(:,1)   = 2*Gzz(:,2) - Gzz(:,3);%(2*G(:,1)    - 5*G(:,2)     + 4*G(:,3)     - G(:,4))    /(h^2);%Gzz(:,2);
Gz(:,end)  = 2*Gz(:,end-1) - Gz(:,end-2);%(-3*G(:,end) + 4*G(:,end-1) - G(:,end-2))/(2*h);%Gz(:,end-1);
Gzz(:,end) = 2*Gzz(:,end-1) - Gzz(:,end-2);%(2*G(:,end)  - 5*G(:,end-1) + 4*G(:,end-2) - G(:,end-3))/(h^2);%Gzz(:,end-1);

%Send back rhs of G eq:
NLSE3D_AS_STEADY_Gc = zeros(size(G));

NLSE3D_AS_STEADY_Gc(2:end,:) = -(OM + c.^2./(4.*a) + (a.*m.^2)./P2(2:end,:) - (R(2:end,:)-d).*((m.*c)./P2(2:end,:))).*G(2:end,:) + ...
                            a.*(Gr(2:end,:).*(1./R(2:end,:)) + Grr(2:end,:) + Gzz(2:end,:)) + s.*G(2:end,:).^2.*conj(G(2:end,:)) +...
                    1i*(-G(2:end,:).*((a*m*Z(2:end,:))./(R(2:end,:).*P2(2:end,:))) - Gr(2:end,:).*((2*a*m*Z(2:end,:))./P2(2:end,:)) - c*Gz(2:end,:) + Gz(2:end,:).*((2*a*m*(R(2:end,:)-d))./P2(2:end,:)));
 
NLSE3D_AS_STEADY_Gc(1,:)    = 2*NLSE3D_AS_STEADY_Gc(2,:) - NLSE3D_AS_STEADY_Gc(3,:); %All guesses


%Replace singularity NaN with 0:
%NLSE3D_AS_STEADY_Gc(isnan(NLSE3D_AS_STEADY_Gc)) = 0;

%Get rid of NANs by interpolating:   
NLSE3D_AS_STEADY_Gc = inpaint_nans(NLSE3D_AS_STEADY_Gc,2);

NLSE3D_AS_STEADY_Gc = [real(NLSE3D_AS_STEADY_Gc(:)) ; imag(NLSE3D_AS_STEADY_Gc(:))];

clear P2;
clear R;
clear Z;
return;

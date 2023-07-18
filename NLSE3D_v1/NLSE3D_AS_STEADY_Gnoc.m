%NLSE3D_AS_STEADY_Gnoc Steady-state PDE of VR with back-flow absorbed into 
%initial g.
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function NLSE3D_AS_STEADY_Gnoc = NLSE3D_AS_STEADY_Gnoc(G,par)

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
z0  = par(14);

rvec  = rmin:h:rmax;
zvec  = zmin:h:zmax;

[Z,R] = meshgrid(zvec,rvec);
P2    = (R-d).^2 + (Z-z0).^2;
PHI   = atan2((Z-z0),(R-d));

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
LAP      = zeros(sizeg);
%Make interier indicies:
rin      = [2:sizeg(1)-1];
zin      = [2:sizeg(2)-1];

%compute first and second derivatives:
Gr(rin,:) = (G(rin+1,:) - G(rin-1,:))/(2*h);
Gz(:,zin) = (G(:,zin+1) - G(:,zin-1))/(2*h);

Grr(rin,:) = (G(rin+1,:) - 2*G(rin,:) + G(rin-1,:))/(h^2);
Gzz(:,zin) = (G(:,zin+1) - 2*G(:,zin) + G(:,zin-1))/(h^2);

%Formulate internal "Laplacian":
LAP(rin,zin) = -((m.^2)./P2(rin,zin)).*G(rin,zin) + ...
               Gr(rin,zin).*(1./R(rin,zin)) + Grr(rin,zin) + Gzz(rin,zin) + ...
               1i.*(-G(rin,zin).*((m*(Z(rin,zin)-z0))./(R(rin,zin).*P2(rin,zin))) - ...
               Gr(rin,zin).*((2*m*(Z(rin,zin)-z0))./P2(rin,zin)) + Gz(rin,zin).*((2*m*(R(rin,zin)-d))./P2(rin,zin)));
          
%Take care of P2=0 singularity:                
LAP(P2==0) = Gr(P2==0).*(1./d) + Grr(P2==0) + Gzz(P2==0); 

U = G;
%MSD Boundary Condition:
LAP(rin,1)   = ((LAP(rin,2)./U(rin,2)   )      + (1/a)*(s*(U(rin,2).*conj(U(rin,2)) - U(rin,1).*conj(U(rin,1))))).*U(rin,1);
LAP(1,zin)   = ((LAP(2,zin)./U(2,zin)   )      + (1/a)*(s*(U(2,zin).*conj(U(2,zin)) - U(1,zin).*conj(U(1,zin))))).*U(1,zin);
LAP(rin,end) = ((LAP(rin,end-1)./U(rin,end-1)) + (1/a)*(s*(U(rin,end-1).*conj(U(rin,end-1)) - U(rin,end).*conj(U(rin,end))))).*U(rin,end);
LAP(end,zin) = ((LAP(end-1,zin)./U(end-1,zin) )+ (1/a)*(s*(U(end-1,zin).*conj(U(end-1,zin)) - U(end,zin).*conj(U(end,zin))))).*U(end,zin);
%Corners
LAP(1,1)     = ((LAP(2,2)./U(2,2)     )            + (1/a)*(s*(U(2,2).*conj(U(2,2)) - U(1,1).*conj(U(1,1))))).*U(1,1);
LAP(end,1)   = ((LAP(end-1,2)./U(end-1,2)  )       + (1/a)*(s*(U(end-1,2).*conj(U(end-1,2)) - U(end,1).*conj(U(end,1))))).*U(end,1);
LAP(1,end)   = ((LAP(2,end-1)./U(2,end-1)  )       + (1/a)*(s*(U(2,end-1).*conj(U(2,end-1)) - U(1,end).*conj(U(1,end))))).*U(1,end);
LAP(end,end) = ((LAP(end-1,end-1)./U(end-1,end-1) )+ (1/a)*(s*(U(end-1,end-1).*conj(U(end-1,end-1)) - U(end,end).*conj(U(end,end))))).*U(end,end);

%Send back rhs of G eq:
NLSE3D_AS_STEADY_Gnoc = -OM*G + s.*G.^2.*conj(G) + a*LAP;

%Set NaN values to 0:
NLSE3D_AS_STEADY_Gnoc(isnan(NLSE3D_AS_STEADY_Gnoc)) = 0;

%Get rid of NANs by interpolating:   
%NLSE3D_AS_STEADY_Gnoc = inpaint_nans(NLSE3D_AS_STEADY_Gnoc,2);

NLSE3D_AS_STEADY_Gnoc = [real(NLSE3D_AS_STEADY_Gnoc(:)) ; imag(NLSE3D_AS_STEADY_Gnoc(:))];

clear P2;
clear R;
clear Z;
return;

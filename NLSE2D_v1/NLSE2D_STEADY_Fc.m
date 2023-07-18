%NLSE2D_STEADY_Fc Stead-state 2D PDE for finding co-moving solutions with velocity c.
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function NLSE2D_STEADY_Fc = NLSE2D_STEADY_Fc(U,par)

OM  = par(1);
a   = par(2);
s   = par(3);
h   = par(4);
M   = par(5);
N   = par(6);
xmin= par(7);
xmax= par(8);
ymin= par(9);
ymax= par(10);

xvec  = xmin:h:xmax;
yvec  = ymin:h:ymax;
[X,Y] = meshgrid(xvec,yvec);

c = U(end);
U = squeeze(U(1:end-1));

half = floor((length(U))/2);

U = U(1:half) + 1i*U(half + 1:end);

U = reshape(U,M,N);

U = U.*exp(-1i.*Y.*(c/(2*a)));

%Calculate laplacian of interier points:
%Get size of data cube:
sizeu     = size(U);
%Initialize Ut and (temp) Laplacian cubes:
LAP       = zeros(sizeu);
%Make interier indicies:
xin       = [2:sizeu(1)-1];
yin       = [2:sizeu(2)-1];
%Compute 2D second order CD Laplacian:
LAP(xin,yin) = (1/h^2)*(U(xin+1,yin) + U(xin-1,yin) +...
                        U(xin,yin+1) + U(xin,yin-1) -...
                      4*U(xin,yin));  
                 
% %Boundary Condition:
LAP(xin,1)   = (LAP(xin,2)./U(xin,2)         + (1/a)*(s*(U(xin,2).*conj(U(xin,2)) - U(xin,1).*conj(U(xin,1))))).*U(xin,1);
LAP(1,yin)   = (LAP(2,yin)./U(2,yin)         + (1/a)*(s*(U(2,yin).*conj(U(2,yin)) - U(1,yin).*conj(U(1,yin))))).*U(1,yin);
LAP(xin,end) = (LAP(xin,end-1)./U(xin,end-1) + (1/a)*(s*(U(xin,end-1).*conj(U(xin,end-1)) - U(xin,end).*conj(U(xin,end))))).*U(xin,end);
LAP(end,yin) = (LAP(end-1,yin)./U(end-1,yin) + (1/a)*(s*(U(end-1,yin).*conj(U(end-1,yin)) - U(end,yin).*conj(U(end,yin))))).*U(end,yin);
%Corners
LAP(1,1)     = (LAP(2,2)./U(2,2)                 + (1/a)*(s*(U(2,2).*conj(U(2,2)) - U(1,1).*conj(U(1,1))))).*U(1,1);
LAP(end,1)   = (LAP(end-1,2)./U(end-1,2)         + (1/a)*(s*(U(end-1,2).*conj(U(end-1,2)) - U(end,1).*conj(U(end,1))))).*U(end,1);
LAP(1,end)   = (LAP(2,end-1)./U(2,end-1)         + (1/a)*(s*(U(2,end-1).*conj(U(2,end-1)) - U(1,end).*conj(U(1,end))))).*U(1,end);
LAP(end,end) = (LAP(end-1,end-1)./U(end-1,end-1) + (1/a)*(s*(U(end-1,end-1).*conj(U(end-1,end-1)) - U(end,end).*conj(U(end,end))))).*U(end,end);
  

%Send back F:
NLSE2D_STEADY_Fc = -OM*U + a*LAP + s.*U.^2.*conj(U);
NLSE2D_STEADY_Fc = [real(NLSE2D_STEADY_Fc(:)) ; imag(NLSE2D_STEADY_Fc(:))];

return;

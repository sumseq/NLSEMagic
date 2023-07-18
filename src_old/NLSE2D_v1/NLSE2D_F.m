%NLSE2D_f  2D Script NLSE Integrator
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function UT = NLSE2D_F(U,V,h,s,a,method,BC,l_a,l_h2,a_12,a_6h2)

%Get size of data cube:
sizeu = size(U);

%Initialize Ut and (temp) Laplacian cubes:
UT  = zeros(sizeu);
LAP = zeros(sizeu);

%Make interier indicies:
xin       = [2:sizeu(1)-1];
yin       = [2:sizeu(2)-1];

%Compute 2D second order CD Laplacian:
LAP(xin,yin) = l_h2*(U(xin+1,yin) + U(xin-1,yin) +...
                     U(xin,yin+1) + U(xin,yin-1) -...
                            4*U(xin,yin));


%Compute boundary conditions on Laplacian:
if(BC==1) %Dirichlet
    %Planes, Edges, and Corners
    LAP(1  ,:) = -l_a*(s.*U(1,:).*  conj(U(1,:))  -V(1,:)).*U(1,:);
    LAP(end,:) = -l_a*(s.*U(end,:).*conj(U(end,:))-V(end,:)).*U(end,:);
    %Dont repeat X-borders:
    LAP(xin,1)   = -l_a*(s.*U(xin,1).*conj(U(xin,1)) - V(xin,1)).*U(xin,1);
    LAP(xin,end) = -l_a*(s.*U(xin,end).*conj(U(xin,end)) - V(xin,end)).*U(xin,end);
elseif(BC==2) %Modulus-Squared Dirichlet
    LAP(xin,1)   = (LAP(xin,2)./U(xin,2)         + l_a*(V(xin,1)-V(xin,2) + s*(U(xin,2).*conj(U(xin,2)) - U(xin,1).*conj(U(xin,1))))).*U(xin,1);
    LAP(1,yin)   = (LAP(2,yin)./U(2,yin)         + l_a*(V(1,yin)-V(2,yin) + s*(U(2,yin).*conj(U(2,yin)) - U(1,yin).*conj(U(1,yin))))).*U(1,yin);
    LAP(xin,end) = (LAP(xin,end-1)./U(xin,end-1) + l_a*(V(xin,end)-V(xin,end-1) + s*(U(xin,end-1).*conj(U(xin,end-1)) - U(xin,end).*conj(U(xin,end))))).*U(xin,end);
    LAP(end,yin) = (LAP(end-1,yin)./U(end-1,yin) + l_a*(V(end,yin)-V(end-1,yin) + s*(U(end-1,yin).*conj(U(end-1,yin)) - U(end,yin).*conj(U(end,yin))))).*U(end,yin);
    %Corners
    LAP(1,1)     = (LAP(2,2)./U(2,2)                 + l_a*(V(1,1)-V(2,2)       + s*(U(2,2).*conj(U(2,2)) - U(1,1).*conj(U(1,1))))).*U(1,1);
    LAP(end,1)   = (LAP(end-1,2)./U(end-1,2)         + l_a*(V(end,1)-V(end-1,2) + s*(U(end-1,2).*conj(U(end-1,2)) - U(end,1).*conj(U(end,1))))).*U(end,1);
    LAP(1,end)   = (LAP(2,end-1)./U(2,end-1)         + l_a*(V(1,end)-V(2,end-1) + s*(U(2,end-1).*conj(U(2,end-1)) - U(1,end).*conj(U(1,end))))).*U(1,end);
    LAP(end,end) = (LAP(end-1,end-1)./U(end-1,end-1) + l_a*(V(end,end)-V(end-1,end-1) + s*(U(end-1,end-1).*conj(U(end-1,end-1)) - U(end,end).*conj(U(end,end))))).*U(end,end);
elseif(BC==3) %LAP=0
    %Since LAP initialized to all zeros, this is alredy taken care of.
end

if(method==11 || method==1)  %CD
   UT(xin,yin) = 1i*(a*LAP(xin,yin) + (s*U(xin,yin).*conj(U(xin,yin)) - V(xin,yin)).*U(xin,yin));
elseif(method==12 || method==2)%2SHOC
   UT(xin,yin) = 1i*(-a_12*(LAP(xin+1,yin) + LAP(xin-1,yin) +...
                            LAP(xin,yin+1) + LAP(xin,yin-1) -...
                         12*LAP(xin,yin)) + ...
                      a_6h2*(U(xin+1,yin+1) + U(xin+1,yin-1) +...
                             U(xin-1,yin+1) + U(xin-1,yin-1) -...
                           4*U(xin,yin)) + ...
             (s*U(xin,yin).*conj(U(xin,yin)) - V(xin,yin)).*U(xin,yin));
end

%Now do boundary conditions on Ut:
if(BC==1) %Dirichlet
    UT(1,:)     = 0;
    UT(end,:)   = 0;
    UT(xin,1)   = 0;
    UT(xin,end) = 0;
elseif(BC==2) %MSD
    UT(xin,1)   = 1i*imag(UT(xin,2)./U(xin,2)).*U(xin,1);
    UT(1,yin)   = 1i*imag(UT(2,yin)./U(2,yin)).*U(1,yin);
    UT(xin,end) = 1i*imag(UT(xin,end-1)./U(xin,end-1)).*U(xin,end);
    UT(end,yin) = 1i*imag(UT(end-1,yin)./U(end-1,yin)).*U(end,yin);
    %Corners
    UT(1,1)     = 1i*imag(UT(2,2)./U(2,2)).*U(1,1);
    UT(end,1)   = 1i*imag(UT(end-1,2)./U(end-1,2)).*U(end,1);
    UT(1,end)   = 1i*imag(UT(2,end-1)./U(2,end-1)).*U(1,end);
    UT(end,end) = 1i*imag(UT(end-1,end-1)./U(end-1,end-1)).*U(end,end);
elseif(BC==3) %Lap=0
    UT(1,:)     = 1i*(s.*U(1,:).*conj(U(1,:))         - V(1,:)    ).*U(1,:);
    UT(end,:)   = 1i*(s.*U(end,:).*conj(U(end,:))     - V(end,:)  ).*U(end,:);
    UT(xin,1)   = 1i*(s.*U(xin,1).*conj(U(xin,1))     - V(xin,1)  ).*U(xin,1);
    UT(xin,end) = 1i*(s.*U(xin,end).*conj(U(xin,end)) - V(xin,end)).*U(xin,end);    
end

clear LAP;
return;

%NLSE3D_f  3D Script NLSE Integrator
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function UT = NLSE3D_F(U,V,h,s,a,method,BC,l_h2,a_12,a_6h2,l_a)

%Get size of data cube:
sizeu = size(U);

%Initialize Ut and (temp) Laplacian cubes:
UT  = zeros(sizeu);
LAP = zeros(sizeu);

%Make interier indicies:
zin       = [2:sizeu(1)-1];
yin       = [2:sizeu(2)-1];
xin       = [2:sizeu(3)-1];

%Compute 3D second order CD Laplacian:
LAP(zin,yin,xin) = l_h2*(U(zin+1,yin,xin) + U(zin-1,yin,xin) +...
                         U(zin,yin+1,xin) + U(zin,yin-1,xin) +...
                         U(zin,yin,xin+1) + U(zin,yin,xin-1) -...
                      6.*U(zin,yin,xin));

if(BC==1) %Dirichlet
    %Planes, Edges, and Corners
    LAP(1  ,:,:) = l_a*(V(1,:,:)   - s.*U(1,:,:).*conj(U(1,:,:))).*U(1,:,:);
    LAP(end,:,:) = l_a*(V(end,:,:) - s.*U(end,:,:).*conj(U(end,:,:))).*U(end,:,:);
    %Dont repeat Y-borders: 
    LAP(zin,1,:)   = l_a*(V(zin,1,:)   - s.*U(zin,1,:).*conj(U(zin,1,:))).*U(zin,1,:);
    LAP(zin,end,:) = l_a*(V(zin,end,:) - s.*U(zin,end,:).*conj(U(zin,end,:))).*U(zin,end,:);
    %Now dont repeat Y or X boundaries:
    LAP(zin,yin,1)   = l_a*(V(zin,yin,1)   - s.*U(zin,yin,1).*conj(U(zin,yin,1))).*U(zin,yin,1);
    LAP(zin,yin,end) = l_a*(V(zin,yin,end) - s.*U(zin,yin,end).*conj(U(zin,yin,end))).*U(zin,yin,end);
elseif(BC==2) %Modulus-Squared Dirichlet
    %6 Planes
    LAP(1,yin,xin)   = (imag(1i*LAP(2,yin,xin)./U(2,yin,xin)) - l_a*(V(2,yin,xin)-V(1,yin,xin) - s*(U(2,yin,xin).*conj(U(2,yin,xin)) - U(1,yin,xin).*conj(U(1,yin,xin))))).*U(1,yin,xin);
    LAP(zin,1,xin)   = (imag(1i*LAP(zin,2,xin)./U(zin,2,xin)) - l_a*(V(zin,2,xin)-V(zin,1,xin) - s*(U(zin,2,xin).*conj(U(zin,2,xin)) - U(zin,1,xin).*conj(U(zin,1,xin))))).*U(zin,1,xin);    
    LAP(zin,yin,1)   = (imag(1i*LAP(zin,yin,2)./U(zin,yin,2)) - l_a*(V(zin,yin,2)-V(zin,yin,1) - s*(U(zin,yin,2).*conj(U(zin,yin,2)) - U(zin,yin,1).*conj(U(zin,yin,1))))).*U(zin,yin,1);    
    LAP(end,yin,xin) = (imag(1i*LAP(end-1,yin,xin)./U(end-1,yin,xin)) - l_a*(V(end-1,yin,xin)-V(end,yin,xin) - s*(U(end-1,yin,xin).*conj(U(end-1,yin,xin)) - U(end,yin,xin).*conj(U(end,yin,xin))))).*U(end,yin,xin);
    LAP(zin,end,xin) = (imag(1i*LAP(zin,end-1,xin)./U(zin,end-1,xin)) - l_a*(V(zin,end-1,xin)-V(zin,end,xin) - s*(U(zin,end-1,xin).*conj(U(zin,end-1,xin)) - U(zin,end,xin).*conj(U(zin,end,xin))))).*U(zin,end,xin);    
    LAP(zin,yin,end) = (imag(1i*LAP(zin,yin,end-1)./U(zin,yin,end-1)) - l_a*(V(zin,yin,end-1)-V(zin,yin,end) - s*(U(zin,yin,end-1).*conj(U(zin,yin,end-1)) - U(zin,yin,end).*conj(U(zin,yin,end))))).*U(zin,yin,end);    
    %12 Edges
    LAP(1,1,    xin) = (imag(1i*LAP(2,2,xin)./U(2,2,xin))         - l_a*(V(2,2,xin)-V(1,1,xin)         - s*(U(2,2,xin).*conj(U(2,2,xin)) - U(1,1,xin).*conj(U(1,1,xin))))).*U(1,1,xin);
    LAP(1,end,  xin) = (imag(1i*LAP(2,end-1,xin)./U(2,end-1,xin)) - l_a*(V(2,end-1,  xin)-V(1,end,xin) - s*(U(2,end-1,  xin).*conj(U(2,end-1,  xin)) - U(1,end,  xin).*conj(U(1,end,  xin))))).*U(1,end,  xin);
    LAP(end,1,  xin) = (imag(1i*LAP(end-1,2,xin)./U(end-1,2,xin)) - l_a*(V(end-1,2,xin)-V(end,1,xin)   - s*(U(end-1,2,xin).*conj(U(end-1,2,xin)) - U(end,1,xin).*conj(U(end,1,xin))))).*U(end,1,xin);
    LAP(end,end,xin) = (imag(1i*LAP(end-1,end-1,xin)./U(end-1,end-1,xin)) + l_a*(V(end-1,end-1,xin)-V(end,end,xin) - s*(U(end-1,end-1,xin).*conj(U(end-1,end-1,xin)) - U(end,end,xin).*conj(U(end,end,xin))))).*U(end,end,xin);
    LAP(zin,1,1)     = (imag(1i*LAP(zin,2,2)./U(zin,2,2))         - l_a*(V(zin,2,2)-V(zin,1,1) - s*(U(zin,2,2).*conj(U(zin,2,2)) - U(zin,1,1).*conj(U(zin,1,1))))).*U(zin,1,1);
    LAP(zin,1,end)   = (imag(1i*LAP(zin,2,end-1)./U(zin,2,end-1)) - l_a*(V(zin,2,end-1)-V(zin,1,end) - s*(U(zin,2,end-1).*conj(U(zin,2,end-1)) - U(zin,1,end).*conj(U(zin,1,end))))).*U(zin,1,end);
    LAP(zin,end,1)   = (imag(1i*LAP(zin,end-1,2)./U(zin,end-1,2)) - l_a*(V(zin,end-1,2)-V(zin,end,1) - s*(U(zin,end-1,2).*conj(U(zin,end-1,2)) - U(zin,end,1).*conj(U(zin,end,1))))).*U(zin,end,1);
    LAP(zin,end,end) = (imag(1i*LAP(zin,end-1,end-1)./U(zin,end-1,end-1)) + l_a*(V(zin,end-1,end-1)-V(zin,end,end) - s*(U(zin,end-1,end-1).*conj(U(zin,end-1,end-1)) - U(zin,end,end).*conj(U(zin,end,end))))).*U(zin,end,end);
    LAP(1,  yin,1)   = (imag(1i*LAP(2,yin,2)./U(2,yin,2))         - l_a*(V(2,yin,2)-V(1,yin,1) - s*(U(2,yin,2).*conj(U(2,yin,2)) - U(1,yin,1).*conj(U(1,yin,1))))).*U(1,yin,1);
    LAP(1,  yin,end) = (imag(1i*LAP(2,yin,end-1)./U(2,yin,end-1)) - l_a*(V(2,yin,end-1)-V(1,yin,end) - s*(U(2,yin,end-1).*conj(U(2,yin,end-1)) - U(1,yin,end).*conj(U(1,yin,end))))).*U(1,yin,end);
    LAP(end,yin,1)   = (imag(1i*LAP(end-1,yin,2)./U(end-1,yin,2)) - l_a*(V(end-1,yin,2)-V(end,yin,1) - s*(U(end-1,yin,2).*conj(U(end-1,yin,2)) - U(end,yin,1).*conj(U(end,yin,1))))).*U(end,yin,1);
    LAP(end,yin,end) = (imag(1i*LAP(end-1,yin,end-1)./U(end-1,yin,end-1)) - l_a*(V(end-1,yin,end-1)-V(end,yin,end) - s*(U(end-1,yin,end-1).*conj(U(end-1,yin,end-1)) - U(end,yin,end).*conj(U(end,yin,end))))).*U(end,yin,end);
    %8 Corners
    LAP(1,1,1)       = (imag(1i*LAP(2,2,2)./U(2,2,2))         - l_a*(V(2,2,2)    -V(1,1,1)   - s*(U(2,2,2).*conj(U(2,2,2))         - U(1,1,1).*conj(U(1,1,1))))).*U(1,1,1);
    LAP(end,1,1)     = (imag(1i*LAP(end-1,2,2)./U(end-1,2,2)) - l_a*(V(end-1,2,2)-V(end,1,1) - s*(U(end-1,2,2).*conj(U(end-1,2,2)) - U(end,1,1).*conj(U(end,1,1))))).*U(end,1,1);
    LAP(1,end,1)     = (imag(1i*LAP(2,end-1,2)./U(2,end-1,2)) - l_a*(V(2,end-1,2)-V(1,end,1) - s*(U(2,end-1,2).*conj(U(2,end-1,2)) - U(1,end,1).*conj(U(1,end,1))))).*U(1,end,1);
    LAP(1,1,end)     = (imag(1i*LAP(2,2,end-1)./U(2,2,end-1)) - l_a*(V(2,2,end-1)-V(1,1,end) - s*(U(2,2,end-1).*conj(U(2,2,end-1)) - U(1,1,end).*conj(U(1,1,end))))).*U(1,1,end);
    LAP(end,end,end) = (imag(1i*LAP(end-1,end-1,end-1)./U(end-1,end-1,end-1)) - l_a*(V(end-1,end-1,end-1)-V(end,end,end) - s*(U(end-1,end-1,end-1).*conj(U(end-1,end-1,end-1)) - U(end,end,end).*conj(U(end,end,end))))).*U(end,end,end);
    LAP(end,end,1)   = (imag(1i*LAP(end-1,end-1,2)./U(end-1,end-1,2))         - l_a*(V(end-1,end-1,2)    -V(end,end,1)  - s*(U(end-1,end-1,2).*conj(U(end-1,end-1,2))         - U(end,end,1).*conj(U(end,end,1))))).*U(end,end,1);
    LAP(1,end,end)   = (imag(1i*LAP(2,end-1,end-1)./U(2,end-1,end-1))         - l_a*(V(2,end-1,end-1)    -V(1,end,end)   - s*(U(2,end-1,end-1).*conj(U(2,end-1,end-1))         - U(1,end,end).*conj(U(1,end,end))))).*U(1,end,end);
    LAP(end,1,end)   = (imag(1i*LAP(end-1,2,end-1)./U(end-1,2,end-1))         - l_a*(V(end-1,2,end-1)    -V(end,1,end)  - s*(U(end-1,2,end-1).*conj(U(end-1,2,end-1))         - U(end,1,end).*conj(U(end,1,end))))).*U(end,1,end);
elseif(BC==3) %LAP=0
    %Since LAP initialized to all zeros, this is alredy taken care of.
end
    

if(method==11 || method==1) %Second order CD method:
    UT(zin,yin,xin) = 1i*(a*LAP(zin,yin,xin) - (V(zin,yin,xin) -...
                          s*U(zin,yin,xin).*conj(U(zin,yin,xin))).*U(zin,yin,xin));   
elseif(method==12 || method==2) %2SHOC
   UT(zin,yin,xin) = 1i*(...
         -a_12*(LAP(zin+1,yin,xin) + LAP(zin,yin+1,xin) + LAP(zin,yin,xin+1) +...
                LAP(zin-1,yin,xin) + LAP(zin,yin-1,xin) + LAP(zin,yin,xin-1) - ...
             10*LAP(zin,yin,xin))  + ...
       a_6h2*(U(zin+1,yin+1,xin) + U(zin,yin+1,xin+1) + U(zin+1,yin,xin+1) +...
              U(zin-1,yin-1,xin) + U(zin,yin-1,xin-1) + U(zin-1,yin,xin-1) +...
              U(zin+1,yin-1,xin) + U(zin,yin+1,xin-1) + U(zin+1,yin,xin-1) +...
              U(zin-1,yin+1,xin) + U(zin,yin-1,xin+1) + U(zin-1,yin,xin+1) -...
           12*U(zin,yin,xin))   - (V(zin,yin,xin) -...
            s*U(zin,yin,xin).*conj(U(zin,yin,xin))).*U(zin,yin,xin));   
end %Method

%Now do boundary conditions on Ut:
if(BC==1) %Dirichlet
    UT(1,:,:)       = 0;
    UT(end,:,:)     = 0;    
    UT(zin,1,:)     = 0;
    UT(zin,end,:)   = 0;    
    UT(zin,yin,1)   = 0;   
    UT(zin,yin,end) = 0;
elseif(BC==2) %Modulus-Squared Dirichlet    
    %6 Planes
    UT(1,yin,xin)   = 1i*imag(UT(2,yin,xin)./U(2,yin,xin)).*U(1,yin,xin);
    UT(zin,1,xin)   = 1i*imag(UT(zin,2,xin)./U(zin,2,xin)).*U(zin,1,xin);    
    UT(zin,yin,1)   = 1i*imag(UT(zin,yin,2)./U(zin,yin,2)).*U(zin,yin,1);    
    UT(end,yin,xin) = 1i*imag(UT(end-1,yin,xin)./U(end-1,yin,xin)).*U(end,yin,xin);
    UT(zin,end,xin) = 1i*imag(UT(zin,end-1,xin)./U(zin,end-1,xin)).*U(zin,end,xin);    
    UT(zin,yin,end) = 1i*imag(UT(zin,yin,end-1)./U(zin,yin,end-1)).*U(zin,yin,end); 
    %12 Edges
    UT(1,1,    xin) = 1i*imag(UT(2,2,xin)./U(2,2,xin)).*U(1,1,xin);
    UT(1,end,  xin) = 1i*imag(UT(2,end-1,xin)./U(2,end-1,xin)).*U(1,end,  xin);
    UT(end,1,  xin) = 1i*imag(UT(end-1,2,xin)./U(end-1,2,xin)).*U(end,1,xin);
    UT(end,end,xin) = 1i*imag(UT(end-1,end-1,xin)./U(end-1,end-1,xin)).*U(end,end,xin);
    UT(zin,1,1)     = 1i*imag(UT(zin,2,2)./U(zin,2,2)).*U(zin,1,1);
    UT(zin,1,end)   = 1i*imag(UT(zin,2,end-1)./U(zin,2,end-1)).*U(zin,1,end);
    UT(zin,end,1)   = 1i*imag(UT(zin,end-1,2)./U(zin,end-1,2)).*U(zin,end,1);
    UT(zin,end,end) = 1i*imag(UT(zin,end-1,end-1)./U(zin,end-1,end-1)).*U(zin,end,end);
    UT(1,  yin,1)   = 1i*imag(UT(2,yin,2)./U(2,yin,2)).*U(1,yin,1);
    UT(1,  yin,end) = 1i*imag(UT(2,yin,end-1)./U(2,yin,end-1)).*U(1,yin,end);
    UT(end,yin,1)   = 1i*imag(UT(end-1,yin,2)./U(end-1,yin,2)).*U(end,yin,1);
    UT(end,yin,end) = 1i*imag(UT(end-1,yin,end-1)./U(end-1,yin,end-1)).*U(end,yin,end);
    %8 Corners
    UT(1,1,1)       = 1i*imag(UT(2,2,2)./U(2,2,2)).*U(1,1,1);
    UT(end,1,1)     = 1i*imag(UT(end-1,2,2)./U(end-1,2,2)).*U(end,1,1);
    UT(1,end,1)     = 1i*imag(UT(2,end-1,2)./U(2,end-1,2)).*U(1,end,1);
    UT(1,1,end)     = 1i*imag(UT(2,2,end-1)./U(2,2,end-1)).*U(1,1,2);
    UT(end,end,end) = 1i*imag(UT(end-1,end-1,end-1)./U(end-1,end-1,end-1)).*U(end,end,end);
    UT(end,end,1)   = 1i*imag(UT(end-1,end-1,2)./U(end-1,end-1,2)).*U(end,end,1);
    UT(1,end,end)   = 1i*imag(UT(2,end-1,end-1)./U(2,end-1,end-1)).*U(1,end,end);
    UT(end,1,end)   = 1i*imag(UT(end-1,2,end-1)./U(end-1,2,end-1)).*U(end,1,end);
elseif(BC==3) %Lap=0    
    UT(1,:,:)       = 1i*(s.*U(1,:,:).*conj(U(1,:,:))             - V(1,:,:) ).*U(1,:,:);
    UT(end,:,:)     = 1i*(s.*U(end,:,:).*conj(U(end,:,:))         - V(end,:,:)).*U(end,:,:);    
    UT(zin,1,:)     = 1i*(s.*U(zin,1,:).*conj(U(zin,1,:))         - V(zin,1,:)).*U(zin,1,:);
    UT(zin,end,:)   = 1i*(s.*U(zin,end,:).*conj(U(zin,end,:))     - V(zin,end,:)).*U(zin,end,:);    
    UT(zin,yin,1)   = 1i*(s.*U(zin,yin,1).*conj(U(zin,yin,1))     - V(zin,yin,1) ).*U(zin,yin,1);   
    UT(zin,yin,end) = 1i*(s.*U(zin,yin,end).*conj(U(zin,yin,end)) - V(zin,yin,end)).*U(zin,yin,end); 
end
     
clear LAP;
return;

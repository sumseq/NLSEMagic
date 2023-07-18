%NLSE1D_f  1D Script NLSE Integrator
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function UT = NLSE1D_F(U,V,h,s,a,method,BC,l_a,l_h2,l76,l_12,par)

%Get size of data cube:
sizeu = size(U);

%Initialize Ut and (temp) Laplacian cubes:
UT  = zeros(sizeu);
LAP = zeros(sizeu);

%Make interier indicies:
xin = (2:sizeu(1)-1);

%Compute 1D second order CD Laplacian:
LAP(xin) = l_h2*(U(xin+1) - 2*U(xin) + U(xin-1));

%Compute boundary conditions on Laplacian:
if(BC==1)     %Dirichlet
      LAP(1)   = l_a*(V(1)   - s.*U(1).*conj(U(1))).*U(1);
      LAP(end) = l_a*(V(end) - s.*U(end).*conj(U(end))).*U(end);
elseif(BC==2) %Modulus-Squared Dirichlet
      LAP(1)   = (imag(1i*LAP(2)./U(2))         + l_a*(V(1)-V(2)       + s*(U(2).*conj(U(2)) - U(1).*conj(U(1))))).*U(1);
      LAP(end) = (imag(1i*LAP(end-1)./U(end-1)) + l_a*(V(end)-V(end-1) + s*(U(end-1).*conj(U(end-1)) - U(end).*conj(U(end))))).*U(end);
elseif(BC==3) %LAP=0
      LAP(1)   = 0;
      LAP(end) = 0;
elseif(BC==4) %One-Sided Diff
      LAP(1)    = (-U(4) + 4*U(3) -5*U(2) + 2*U(1))/(h^2);    
      LAP(end)  = (-U(end-3) + 4*U(end-2) -5*U(end-1) + 2*U(end))/(h^2);
elseif(BC==5) %Exact for dark soliton (IC2 only)
      t  = par(1);
      x1 = par(2);
      xn = par(3);
      OM = par(4);
      c  = par(5);
      x = x1;
      LAP(1)   = (exp(1i*(t*(OM - c^2/(4*a)) + (c*x)/(2*a)))*(OM/s)^(1/2)*(4*a^2*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))^3*abs(OM) - 4*a^2*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))*abs(OM) + c^2*1i^2*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))*abs(a) + 4*a*c*1i*abs(a)^(1/2)*(abs(OM)/2)^(1/2) - 4*a*c*1i*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))^2*abs(a)^(1/2)*(abs(OM)/2)^(1/2)))/(4*a^2*abs(a));
      x = xn;
      LAP(end) = (exp(1i*(t*(OM - c^2/(4*a)) + (c*x)/(2*a)))*(OM/s)^(1/2)*(4*a^2*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))^3*abs(OM) - 4*a^2*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))*abs(OM) + c^2*1i^2*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))*abs(a) + 4*a*c*1i*abs(a)^(1/2)*(abs(OM)/2)^(1/2) - 4*a*c*1i*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))^2*abs(a)^(1/2)*(abs(OM)/2)^(1/2)))/(4*a^2*abs(a));
end

if(method==11 || method==1)    %CD
   UT(xin) = 1i*(a*LAP(xin) + (s*U(xin).*conj(U(xin)) - V(xin)).*U(xin));
elseif(method==12 || method==2)%2SHOC
   UT(xin) = 1i*(a*(l76*LAP(xin) - l_12*(LAP(xin+1) + LAP(xin-1))) + (s*U(xin).*conj(U(xin)) - V(xin)).*U(xin)); 
end

%Now do boundary conditions on Ut:
if(BC==1) %Dirichlet
       UT(1)   = 0;
       UT(end) = 0;
elseif(BC==2) %MSD
       UT(1)   = 1i*imag(UT(2)./U(2))*U(1);  
       UT(end) = 1i*imag(UT(end-1)./U(end-1))*U(end);
elseif(BC==3) %Uxx=0
       UT(1)   = 1i*(s.*U(1).*conj(U(1)) - V(1)).*U(1);
       UT(end) = 1i*(s.*U(end).*conj(U(end)) - V(end)).*U(end);
elseif(BC==4) %One-sided Difference (2SHOC)
       dx41    = (-LAP(4) + 4*LAP(3) -5*LAP(2) + 2*LAP(1))/(h^2);    
       dx4e    = (-LAP(end-3) + 4*LAP(end-2) -5*LAP(end-1) + 2*LAP(end))/(h^2);   
       UT(1)   = 1i*(a*(LAP(1)   - (h^2/12)*dx41) + (s.*U(1).*conj(U(1))     -   V(1)).*U(1));   
       UT(end) = 1i*(a*(LAP(end) - (h^2/12)*dx4e) + (s.*U(end).*conj(U(end)) - V(end)).*U(end));  
elseif(BC==5) %Exact for dark soliton
       t  = par(1);
       x1 = par(2);
       xn = par(3);
       OM = par(4);
       c  = par(5);
       x = x1;  
       UT(1)   = 1i*exp(1i*(t*(OM - c^2/(4*a)) + (c*x)/(2*a)))*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))*(OM/s)^(1/2)*(OM - c^2/(4*a)) + (c*exp(1i*(t*(OM - c^2/(4*a)) + (c*x)/(2*a)))*(abs(OM)/2)^(1/2)*(tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))^2 - 1)*(OM/s)^(1/2))/abs(a)^(1/2);
       x = xn;
       UT(end) = 1i*exp(1i*(t*(OM - c^2/(4*a)) + (c*x)/(2*a)))*tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))*(OM/s)^(1/2)*(OM - c^2/(4*a)) + (c*exp(1i*(t*(OM - c^2/(4*a)) + (c*x)/(2*a)))*(abs(OM)/2)^(1/2)*(tanh(((x - c*t)*(abs(OM)/2)^(1/2))/abs(a)^(1/2))^2 - 1)*(OM/s)^(1/2))/abs(a)^(1/2);
end 

clear LAP;
return

%NLS_add_vortex  Routine to add a vortex to the current solution.
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function U = NLS_add_vortex(charge,om,x0,y0,vx,vy,phase,U,h,a,s,amp,X,Y,opt)

plot_profiles = 0;
m             = charge;
OM            = om;

R     = sqrt((X-x0).^2+(Y-y0).^2);
Theta = atan2((Y-y0),(X-x0));

rh = h;

rmax    = max(R(:));
rvec    = (0:rh:rmax+rh)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%INITIAL CONDITION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set up ansatz:
if(sign(s)==-1)
    OM = amp^2*s;    
    if(opt==0)
       Finitv = sqrt(OM/s).*tanh(sqrt(abs(OM/(2*a))).*(rvec)); 
    else
       Finitv = real(sqrt(OM/s + a*m^2./(s*rvec.^2)));
    end
    lBC = 0;
    rBC = sqrt(OM/s + a*m^2./(s*max(rvec).^2));
elseif(sign(s)==1)
    B      = sqrt((3*OM)/s);
    C      = sqrt((3*OM)/(2*a));
    R0     = sqrt((2*m^2*a)/OM); 
    Finitv = B.*sech(C*(rvec-R0)); 
    lBC = 0;
    rBC = 0;
end

Finitv(1)   = lBC;
Finitv(end) = rBC;

if(plot_profiles==1)
    figure(100)
    plot(rvec,Finitv,'b')
    hold on;
end

if(opt==1)
   par = [m,OM,a,s,rh,lBC,rBC];
   maxit         = 40;
   maxitl        = 40;
   etamax        = 0.9;
   lmeth         = 2;
   restart_limit = 20;
   sol_parms     = [maxit,maxitl,etamax,lmeth,restart_limit];
   [Finitv,it_hist,ierr] = nsoli(Finitv,@(Utmp)NLS_STEADY_F(Utmp,par),(1e-8)*[1,1],sol_parms);
   if ierr ~= 0
    disp('Error in optimization, best profile given.');    
   else
    disp(['2D Single Vortex Profile found, adding to IC...']);
   end
end

if(plot_profiles==1)
   figure(100)
   plot(rvec,Finitv,'r');
   title(['f(r) profile for OM=',num2str(OM),' and m=',num2str(m)]);
   legend('Init Guess','GN Sol');
   hold off;    
   figure(101)
   plot(rvec,NLS_STEADY_F(Finitv,par))
   title('RHS Residuals for f(r)');         
   pause    
end   



if((s<0 && abs(m)==1)&& opt==0) %Use tanh ansatz for unoptimized dark vortex of charge 1
      Uinit = sqrt(OM/s).*tanh(sqrt(abs(OM/(2*a))).*R).*exp(1i*(m*Theta + phase)).*exp(1i*vx.*X).*exp(1i*vy.*Y);
else
%Get unique sorted r values of 2D grid:
rv = unique(R(:));
%Interpolate f(r) values for 2D grid using f(r) found above:
Finterpv = interp1(rvec,Finitv,rv); 
    
%Formulate Finit matrix:    
Finit = zeros(size(R));
sizeR = size(R);
 for i=1:sizeR(1)
   for j=1:sizeR(2)
        %Get f(r) value for 2D grid point using interpolated vector:                
        Finit(i,j) = Finterpv(R(i,j)==rv);             
   end %j
 end%i    
Uinit = Finit.*exp(1i*(m*Theta + phase)).*exp(1i*vx.*X).*exp(1i*vy.*Y);
end

%Now add to U:
if(sign(s)==-1)
    %Normalize old and new U:
    ampUinit = max(max(abs(Uinit)));
    Uinit    = Uinit./ampUinit;
    ampU     = max(max(abs(U)));
    U        = U./ampU;
    %Multiply new solution into U:
    U = U.*Uinit;
    %Rescale back to new U amp
    U = U.*ampUinit;
elseif(sign(s)==1)
    %Add new vortex to grid:
    U = U + Uinit;
end

return
end

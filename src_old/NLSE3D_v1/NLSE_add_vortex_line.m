%NLS_add_vortex_line  Routine to add a vortex line along z-axis to the current solution.
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function U = NLSE_add_vortex_line(m,OM,x0,y0,z0,vx,vy,vz,...
                                 phase,U,h,a,s,amp,X,Y,Z)

plot_profiles = 0;
rh = h;

%Make R-theta grid on X-Y plane
R     = squeeze(sqrt((X(:,:,1)-x0).^2+(Y(:,:,1)-y0).^2));
Theta = squeeze(atan2((Y(:,:,1)-y0),(X(:,:,1)-x0)));
rmax    = max(R(:));
rvec    = (0:rh:rmax+rh)';

%Set up 1D ansatz:
if(sign(s)==-1)
    Finitv = real(sqrt(OM/s + a*m^2./(s*rvec.^2)));
    lBC    = 0;
    rBC    = sqrt(OM/s + a*m^2./(s*max(rvec).^2));
elseif(sign(s)==1)
    B      = sqrt((3*OM)/s);
    C      = sqrt((3*OM)/(2*a));
    R0     = sqrt((2*m^2*a)/OM); 
    Finitv = B.*sech(C*(rvec-R0)); 
    lBC    = 0;
    rBC    = 0;
end

Finitv(1)   = lBC;
Finitv(end) = rBC;

if(plot_profiles>1)
   figure(100)
   plot(rvec,Finitv,'--r','LineWidth',3)  
   hold on;
end

par = [m,OM,a,s,rh,lBC,rBC];
maxit         = 50;
maxitl        = 50;
etamax        = 0.9;
lmeth         = 2;
restart_limit = 20;
sol_parms     = [maxit,maxitl,etamax,lmeth,restart_limit];

[Finitv,it_hist,ierr] = nsoli(Finitv,@(Utmp)NLS_STEADY_F(Utmp,par),1e-14*[1,1],sol_parms);

Finit = zeros(size(R)); 

if ierr ~= 0
    disp('Error in optimization!');
    Finit = 0.*R;
else  
    if(plot_profiles>1)
       figure(100)
       hold on;
       plot(rvec,Finitv,'-b','LineWidth',3);
       title(['f(r) profile for OM=',num2str(OM),' and m=',num2str(m)]);
       legend('Init Guess','GN Sol');
       hold off;
       
       figure(11)
       plot(rvec,NLS_STEADY_Fnsoli(Finitv,par))
       title('RHS Residuals for f(r)');         
       pause
    end
   
 %Formulate 2D init matrix:
    %Get unique sorted r values of 2D grid:
    rv = unique(R(:));
    %Interpolate f(r) values for 2D grid using f(r) found above:
    Finterpv = interp1(rvec,Finitv,rv);    
    %Formulate Finit matrix:
    sizeR = size(R);
    for i=1:sizeR(1)
        for j=1:sizeR(2)
          %Get f(r) value for 2D grid point using interpolated vector:                
          Finit(i,j) = Finterpv(R(i,j)==rv);             
        end %j
    end%i    
end %opt worked

%Add phase to 2D densty profile:
Uinit2D = Finit.*exp(1i*(m*Theta + phase));

if(plot_profiles==1)
   figure(111)   
   surf(squeeze(X(:,:,1)-x0),squeeze(Y(:,:,1)-y0),abs(Uinit2D))
   view([0 90]);
end

Uinit3D = zeros(size(U));

%Formulate line along z-axis:
cube_size = size(U);

for zi=1:cube_size(3)
   Uinit3D(:,:,zi) = Uinit2D;
end   

%Now add new VL to U:
if(sign(s)==-1)
    %Normalize old and new U:
    ampU    = max(abs(U(:)));
    ampUnew = max(abs(Uinit3D(:)));
    %Multiply new solution into U:
    U = (U./ampU).*(Uinit3D./ampUnew);
    %Rescale back to new U amp
    U = ampUnew.*U.*exp(1i*(vx*X + vy*Y + vz*Z));
elseif(sign(s)==1)
    %Add new vortex to grid:
    U = U + Uinit3D;
end
clear R;
clear Theta;
clear Uinit3D;
return
end

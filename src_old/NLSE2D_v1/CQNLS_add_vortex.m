function U = CQNLS_add_vortex(charge,om,om_str,x0,y0,vx,vy,phase,U,h,a,s,amp,X,Y)

plot_profiles=0;
m  = charge;
OM = om;
OMstr = om_str;

R     = sqrt((X-x0).^2+(Y-y0).^2);
Theta = atan2((Y-y0),(X-x0));

rh = h;

rmax    = max(R(:));
rvec    = (0:rh:rmax+rh)';

R     = sqrt((X-x0).^2+(Y-y0).^2);
Theta = atan2((Y-y0),(X-x0));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%INITIAL CONDITION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Set up ansatz:



%OMvatest = F_OM_va(OMstr);
%disp(['Error in Omega VA Computation: ',num2str(abs(OM-OMvatest))]);
r0VA      = abs(m)./sqrt(OMstr-OM);
Uva_P     = 4*OMstr;
Uva_T     = sqrt(1-(16/3)*OMstr);
Uva_W     = sqrt(Uva_P);
Finitv    = double(sqrt(Uva_P./(1+ Uva_T.*cosh(Uva_W.*(rvec - r0VA)))));   


lBC = 0;
rBC = 0;

Finitv(1)       = lBC;            
Finitv(end)     = rBC;     

if(plot_profiles==1)
    figure(100)
    plot(rvec,Finitv,'b')
    hold on;
end

   par = [m,OM,a,s,rh,lBC,rBC];
   maxit         = 40;
   maxitl        = 40;
   etamax        = 0.9;
   lmeth         = 2;
   restart_limit = 20;
   sol_parms     = [maxit,maxitl,etamax,lmeth,restart_limit];
   [Finitv,it_hist,ierr] = nsoli(Finitv,@(Utmp)CQNLS_STEADY_F(Utmp,par),(1e-8)*[1,1],sol_parms);
   if ierr ~= 0
    disp('Error in optimization, best profile given.');    
   else
    disp(['2D Single Vortex Profile found, adding to IC...']);
   end
        

    if(plot_profiles==1)
    figure(100)
    plot(rvec,Finitv,'r');
    title(['f(r) profile for OM=',num2str(OM),' and m=',num2str(m)]);
    legend('Init Guess','GN Sol');
    hold off;
    pause
    end

   
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

%Now add to U:
U = U + Uinit;


return
end
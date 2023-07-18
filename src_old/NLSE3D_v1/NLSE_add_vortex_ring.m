%NLS_add_vortex_ring  Routine to add a vortex ring to the current solution.
%©2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

function U = NLSE_add_vortex_ring(m,OM,x0,y0,z0,thetax,thetay,thetaz,vx,vy,vz,...
                                 phase,U,h,a,s,X,Y,Z,d,opt_2D,ea,eb)
plot_profiles = 0;  %Plot and save profiles
optwc         = 0;  %Optimize with c inside steady-state PDE (1) or not (0).
optm          = 2;  %Use initial phase with imT (1) or imT-imT2 (2).

if(s<0)
%Vortex core parameters:
if(abs(m)==1)
   L0 = 0.380876183708681;
elseif(abs(m)==2)
   L0 = 0.133840;
elseif(abs(m)==3)
   L0 = 0.070769;
elseif(abs(m)==4)
   L0 = 0.0445940;
elseif(abs(m)==5)
   L0 = 0.030981;
end

%VR velocity::
c =-((a*m)/d)*(log((8*d)/(abs(m)*sqrt(a/(-OM)))) + L0 - 1);
%Overide c with m=1 c:
%c =-((a*sign(m))/d)*(log((8*d)/sqrt(a/(-OM)))+ L0 - 1);
end

%If making a bright ring, set c=0:
if(s>0); c=0; end;

%This allows one to change the 2D vortex OM
OM1d = OM;

%Set step size for 1D radial profile:
rh = h/2;

%Display what the routine is doing:
if(opt_2D>0)
   opttxt = 'optimized';
else
   opttxt = []; 
end
if(s>0)
   stxt = 'bright';
else
   stxt = 'dark'; 
end
disp(['Vortex Ring Generator (VRG) forming ',opttxt,' ',stxt,' VR with']);
disp([' charge ',num2str(m),' radius ',num2str(d),...
    ', position (',    num2str(x0),',',    num2str(y0),',',    num2str(z0),...
    '), orientation']);
disp([' (',num2str(thetax),',',num2str(thetay),',',num2str(thetaz),...
    '), velocity (',   num2str(vx),',',    num2str(vy),',',    num2str(vz),...
    '), phase ',num2str(phase),'...']);

%Formulate rotation matrices:
Rx = [ 1            0            0;...
       0            cos(thetax) -sin(thetax);...
       0            sin(thetax)  cos(thetax)]; 
Ry = [ cos(thetay)  0            sin(thetay);...
       0            1            0;...
      -sin(thetay)  0            cos(thetay)];
Rz = [cos(thetaz)  -sin(thetaz)  0;...
      sin(thetaz)   cos(thetaz)  0;...
      0             0            1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COMPUTE SIZE AND POSITION OF REQUIRED ROTATED SOLUTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Original grid limits
xmin = min(X(:));
xmax = max(X(:));
ymin = min(Y(:));
ymax = max(Y(:));
zmin = min(Z(:));
zmax = max(Z(:));

%Compute new VR position:
xyz0R = (Rx*Ry*Rz)*[x0;y0;z0];
x0R = xyz0R(1);
y0R = xyz0R(2);
z0R = xyz0R(3);

%Formulate location of grid corner points:
cornerpoints              = zeros(3,8);
cornerpoints(1,1:4)       = xmin;
cornerpoints(1,5:8)       = xmax;
cornerpoints(2,[1,2,5,6]) = ymin;
cornerpoints(2,[3,4,7,8]) = ymax;
cornerpoints(3,[1,3,5,7]) = zmin;
cornerpoints(3,[2,4,6,8]) = zmax;

%Rotate 8 corner points:
for k=1:8
    xyzR = (Rx*Ry*Rz)*[cornerpoints(1,k);cornerpoints(2,k);cornerpoints(3,k)];
    cornerpointsR(1,k) = xyzR(1);
    cornerpointsR(2,k) = xyzR(2);
    cornerpointsR(3,k) = xyzR(3);
end
 
%Now compute required max and mins from rotated corners:
xminR = min(cornerpointsR(1,:));
yminR = min(cornerpointsR(2,:));
zminR = min(cornerpointsR(3,:));
xmaxR = max(cornerpointsR(1,:));
ymaxR = max(cornerpointsR(2,:));
zmaxR = max(cornerpointsR(3,:));

%Formulate new grid limits and create matrices 
xvecR      = xminR:h:xmaxR; 
yvecR      = yminR:h:ymaxR; 
zvecR      = zminR:h:zmaxR; 
[XR,YR,ZR] = meshgrid(xvecR,yvecR,zvecR);
Uinit3DR   = zeros(size(XR));
sizeuR     = size(Uinit3DR);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  NOW COMPUTE VR CUBE ON ROTATED GRID
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute minimum required 2D r-direction about the XY plave grid size:
rxy_max = max([sqrt((x0R-xminR)^2 + (y0R-yminR)^2),...
               sqrt((x0R-xminR)^2 + (y0R-ymaxR)^2),...
               sqrt((x0R-xmaxR)^2 + (y0R-yminR)^2),...
               sqrt((x0R-xmaxR)^2 + (y0R-ymaxR)^2)]);                  

%Compute resolution of 2D r-direction needed:              
rxy_res = ceil(rxy_max/h)+2;
zres    = sizeuR(3);
          
%Create 2D axisymmetric Rxy-Z grid:
Finit = zeros(rxy_res, zres);
    
%Create new Z and RXY mattrices for Finit:
rxy_vec   = [0:h:(rxy_res-1)*h];
[Z2D RXY] = meshgrid(zvecR,rxy_vec);

%Create R and Phi matrices for RXY-Z axisymmetric plane:
RXZ   = sqrt( (Z2D-z0R).^2 + (RXY-d).^2 );
Phi   = atan2((Z2D-z0R),(RXY-d));
Phi2  = atan2((Z2D-z0R),(RXY+d));

%Now create 1D rxz vector for radial profile of 2D vortex
rxz_max  = 2*max(RXZ(:));
rxz_res  = ceil(rxz_max/rh)+2;
rxz_vec  = (0:rh:(rxz_res-1)*rh)';

%Set up 1D radial profile ansatz for 2D vortex:
if(sign(s)==-1)
    Finitv = real(sqrt(OM1d/s + a*m^2./(s*rxz_vec.^2)));
    lBC    = 0;
    rBC    = sqrt(OM1d/s + a*m^2./(s*max(rxz_vec).^2));
elseif(sign(s)==1)
    B      = sqrt((3*OM1d)/s);
    C      = sqrt((3*OM1d)/(2*a));
    R0     = sqrt((2*m^2*a)/OM1d); 
    Finitv = B.*sech(C*(rxz_vec-R0)); 
    lBC    = 0;
    rBC    = 0;
end

%Set up nsoli optimization paramaters:
par = [m,OM1d,a,s,rh,lBC,rBC];
maxit         = 70;
maxitl        = 50;
etamax        = 0.9999;
lmeth         = 2;
restart_limit = 2;
sol_parms     = [maxit,maxitl,etamax,lmeth,restart_limit];

%Set boundary condition values:
Finitv(1)   = lBC;
Finitv(end) = rBC;

%Plot initial profile ansatz:
if(plot_profiles>=1)
   fig_prof = figure(1);  
   hold on;
   if(m==1 && s==-1) %Use tanh for m=1 dark vortex profile (better than sqrt)
      plot(rxz_vec,sqrt(OM/s).*tanh(sqrt(abs(OM/(2*a))).*(rxz_vec)),'-.r','LineWidth',3); 
   end
   plot(rxz_vec,Finitv,'--b','LineWidth',3);   
end

fprintf('VRG:  Begining 1D optimization of vortex radial profile...');
%Optimize profile:
[Finitv,it_hist,ierr] = nsoli(Finitv,@(Utmp)NLS_STEADY_F(Utmp,par),1e-12*[1,1],sol_parms);

if ierr ~= 0
    disp('VRG:  Error in 1D vortex profile optimization!');
    Finit = 0.*RXZ;
else  
    fprintf('Done!\n');
    %Plot optimized profile:
    if(plot_profiles>=1)
       figure(fig_prof)
       hold on;
       plot(rxz_vec,Finitv,'-k','LineWidth',5);
       axis([0 20 0 1.1]);
       set(gca,'FontSize',18);
       xlabel('r');
       ylabel('f(r)');     
       set(fig_prof, 'PaperPositionMode', 'auto'); 
       print('-depsc','-r100','-opengl',['fofr_m',num2str(m),'_OM',num2str(OM)],['-f',num2str(fig_prof)]); 
      % title(['f(r) profile for OM=',num2str(OM1d),' and m=',num2str(m)]);
      % legend('Init Guess','GN Sol');
       hold off;       
       f1d2 = figure(101);
       plot(rxz_vec,NLS_STEADY_F(Finitv,par))
       title('RHS Residuals for f(r)');         
       pause
    end    
    %Formulate 2D initial matrix:
    fprintf('VRB:  Formulating 2D vortex through interpolation...');
    %Get unique sorted r values of 2D grid:
    rxz2D_vec = unique(RXZ(:));
    %Interpolate f(r) values for 2D grid using f(r) found above:
    Finterpv = interp1(rxz_vec,Finitv,rxz2D_vec);    
    %Formulate Finit matrix (r vs z plane):
    sizeRXZ = size(RXZ);
    for i=1:sizeRXZ(1)
        for j=1:sizeRXZ(2)
          %Get f(r) value for 2D grid point using interpolated vector:                
          Finit(i,j) = Finterpv(RXZ(i,j)==rxz2D_vec);             
        end %j
    end%i        
end %opt worked

%Now add vorticity and phase to form 2D vortex with added relfection vorticity:
if(optm==2)
    Uinit2D = Finit.*exp(1i*((m*Phi - m*Phi2) + phase));
elseif(optm==1)
    Uinit2D = Finit.*exp(1i*(m*Phi + phase));    
end
fprintf('Done!\n');

%Plot vortex solution:
if(plot_profiles==2)
   f2d1 = figure(102);
   surf(RXY,Z2D,(Uinit2D.*conj(Uinit2D)));  shading interp;
   view([0 90]);
   axis equal;
   axis([min(RXY(:)) max(RXY(:)) min(Z2D(:)) max(Z2D(:))]);
   drawnow
   set(gca,'FontSize',18);
   xlabel('r');
   ylabel('z');     
   set(f2d1, 'PaperPositionMode', 'auto'); 
   print('-depsc','-r100','-opengl',['Uninit2D_init_optm',num2str(optm),'_m',num2str(m),'_OM',num2str(OM),'_MOD2'],['-f',num2str(f2d1)]); 
   
   f2d1phase = figure(103);
   surf(RXY,Z2D,angle(Uinit2D));  shading interp;
   view([0 90]);
   axis equal;
   axis([min(RXY(:)) max(RXY(:)) min(Z2D(:)) max(Z2D(:))]);
   drawnow
   set(gca,'FontSize',18);
   xlabel('r');
   ylabel('z');     
   set(f2d1phase, 'PaperPositionMode', 'auto'); 
   print('-depsc','-r100','-opengl',['Uninit2D_init_optm',num2str(optm),'_m',num2str(m),'_OM',num2str(OM),'_PHASE'],['-f',num2str(f2d1phase)]); 
end

%Now optimize 2D VR slice is desired:
if(opt_2D>0)           
    sizeu  = size(Uinit2D);  
    M      = sizeu(1);
    N      = sizeu(2);        
    rxy_min = min(RXY(:));    
    rxy_max = max(RXY(:));   
    
    par = [OM a s h M N c d m rxy_min rxy_max zminR zmaxR z0]; 
    
    if(opt_2D==1)      
       if(optwc~=1)          
          Uinit2D = Uinit2D.*exp(1i.*Z2D.*(-c/(2*a)));
       end
       %Flatten out 2D slice into 1D real vector:
       Uinit2D = [real(Uinit2D(:)) ; imag(Uinit2D(:))];
       if(optwc==1)
           rhs1 = NLSE3D_AS_STEADY_Uc(Uinit2D,par);  
       else
           rhs1 = NLSE3D_AS_STEADY_Unoc(Uinit2D,par);   
       end
    end    
    
    if(opt_2D==2)      
       if(optwc==1)
           if(optm==1)
               FinitC = Finit;
           elseif(optm==2) %Add in second interference m not accounted for in geq:
               FinitC = Finit.*exp(-1i*m*Phi2);
           end
       else
           FinitC = Finit.*exp(1i.*Z2D.*(-c/(2*a)));
       end
       %Flatten out 2D slice into 1D real vector:
       FinitV = [real(FinitC(:)) ; imag(FinitC(:))];
       if(optwc==1)
           rhs1 = NLSE3D_AS_STEADY_Gc(FinitV,par);  
       else
           rhs1 = NLSE3D_AS_STEADY_Gnoc(FinitV,par);   
       end
    end    
    disp(['VRG:  Inf-norm Error for initial VR slice: ',num2str(norm(rhs1(:),inf))]);
    
    %Plot initial residuals:
    if(plot_profiles==2)
       rhs1 = rhs1(1:(length(rhs1))/2) + 1i*rhs1(((length(rhs1))/2 + 1):end);
       rhs1 = reshape(rhs1,M,N);
       f2d3 = figure(104);    
       hold on;       
       surf(RXY,Z2D,abs(rhs1));
       hold off;
       xlabel('x');ylabel('z');zlabel('RHS1'); shading interp;    axis tight;
       view([-30 30]);
       drawnow;
       pause;
    end
  
    %Setup nsoli parameters for 2D optimization:
    maxit         = 70;
    maxitl        = 50;
    etamax        = 0.9999;
    lmeth         = 2;
    restart_limit = 2;
    sol_parms=[maxit,maxitl,etamax,lmeth,restart_limit];   
    
    fprintf('VRG:  Starting 2D optimizaion for vortex ring...');  
    if(opt_2D==1)       
       if(optwc==1)
         [Uinit2D,it_hist,ierr] = nsoli(Uinit2D,  @(Utmp)NLSE3D_AS_STEADY_Uc(Utmp,par),  1e-4*[1,1],sol_parms); 
         rhs2 = NLSE3D_AS_STEADY_Uc(Uinit2D,par); 
       else
          [Uinit2D,it_hist,ierr] = nsoli(Uinit2D,  @(Utmp)NLSE3D_AS_STEADY_Unoc(Utmp,par),  1e-5*[1,1],sol_parms);
          rhs2 = NLSE3D_AS_STEADY_Unoc(Uinit2D,par); 
       end       
       Uinit2D = Uinit2D(1:(length(Uinit2D))/2) + 1i*Uinit2D(((length(Uinit2D))/2 + 1):end);
       Uinit2D = reshape(Uinit2D,M,N);        
             
       if(optwc==0)
          Uinit2D = Uinit2D.*exp(1i.*Z2D.*(c/(2*a)));
       end
    end
    
    if(opt_2D==2)                    
       if(optwc==1)
          [FinitV,it_hist,ierr] = nsoli(FinitV,  @(Utmp)NLSE3D_AS_STEADY_Gc(Utmp,par),  1e-4*[1,1],sol_parms);
          rhs2  = NLSE3D_AS_STEADY_Gc(FinitV,par);
       else           
          [FinitV,it_hist,ierr] = nsoli(FinitV,  @(Utmp)NLSE3D_AS_STEADY_Gnoc(Utmp,par),  1e-6*[1,0],sol_parms);
          rhs2  = NLSE3D_AS_STEADY_Gnoc(FinitV,par);
       end         
       FinitV  = FinitV(1:(length(FinitV))/2)  + 1i*FinitV(((length(FinitV))/2 + 1):end);
       Finit   = reshape(FinitV,M,N); 
       if(optwc==1)
           Uinit2D = Finit.*exp(1i*(m*Phi + phase));
       else
           Uinit2D = Finit.*exp(1i*(m*Phi + phase)).*exp(1i.*Z2D.*(c/(2*a)));
       end      
    end         
    fprintf('Done!\n');            
    
    if(ierr ~= 0)
       disp(['VRG: WARNING! 2D vortex ring optimization did not converge!  Error:',num2str(ierr)]);
    end      
        
    disp(['VRG:  Inf-norm Error for optimized VR slice: ',num2str(norm(rhs2(:),inf))]);
    rhs2 = rhs2(1:(length(rhs2))/2) + 1i*rhs2(((length(rhs2))/2 + 1):end);
    if(plot_profiles==2)
       rhs2 = reshape(rhs2,M,N);        
       f2d4 = figure(105);
       hold on;
       surf(RXY,Z2D,abs(rhs2));
       view([-30 30]);
       hold off;
       xlabel('x');ylabel('z');zlabel('RHS2')
       shading interp
       axis tight;    
       drawnow;
    end
end%opt_2D

%Plot optimized VR slice:
if(plot_profiles==2)
   f2d2 = figure(106);
   surf(RXY,Z2D,(Uinit2D.*conj(Uinit2D)));   shading interp;
   view([0 90]);
   axis equal;
   axis([min(RXY(:)) max(RXY(:)) min(Z2D(:)) max(Z2D(:))]);
   drawnow
   set(gca,'FontSize',18);
   xlabel('r');
   ylabel('z');     
   set(f2d2, 'PaperPositionMode', 'auto'); 
   print('-depsc','-r100','-opengl',['Uninit2D_final_optm',num2str(optm),'_m',num2str(m),'_OM',num2str(OM),'_MOD2'],['-f',num2str(f2d2)]); 
   
   f2d2phase = figure(107);
   surf(RXY,Z2D,angle(Uinit2D)); shading interp;
   view([0 90]);
   axis equal;
   axis([min(RXY(:)) max(RXY(:)) min(Z2D(:)) max(Z2D(:))]);
   drawnow
   set(gca,'FontSize',18);
   xlabel('r');
   ylabel('z');     
   set(f2d2phase, 'PaperPositionMode', 'auto'); 
   print('-depsc','-r100','-opengl',['Uninit2D_final_optm',num2str(optm),'_m',num2str(m),'_OM',num2str(OM),'_PHASE'],['-f',num2str(f2d2phase)]); 
pause
end

%Formulate 3D solution from 2D radial slice though interpolation

%Now need to rotate 2D matrix around Z-axis for 3D ring:
RXYreal         = squeeze(sqrt(((XR(:,:,1)-x0R)./ea).^2+((YR(:,:,1)-y0R)./eb).^2));
rxyreal2D_vec   = unique(RXYreal(:));

fprintf('VRB:  Formulating 3D vortex ring through interpolation...')
Finterpv    = interp1(rxy_vec,Uinit2D,rxyreal2D_vec,'linear','extrap');
sizeRXYreal = size(RXYreal);
for i=1:sizeRXYreal(1)
    for j=1:sizeRXYreal(2)
       %Get f(r) value for 3D grid vector using interpolated vector:                
       Uinit3DR(i,j,:) = Finterpv(RXYreal(i,j)==rxyreal2D_vec,:);             
    end %j
end%i    
%Add velocity
Uinit3DR = Uinit3DR.*exp(1i*(vx*XR + vy*YR + vz*ZR));
fprintf('Done!\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOW NEED TO INTERPOLATE INTO ORIGINAL GRID WITH ROTATION:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Uinit3D = zeros(size(X));
sizeu   = size(Uinit3D);
sizeuR  = size(Uinit3DR); 

if(((thetax==pi/2 || thetax==0) && (thetay==pi/2 || thetay==0)) && (thetaz==pi/2 || thetaz==0))
    if(thetax==pi/2)  %x-90 transpose
    %Do a simple transpose for speed:
    Uinit3DR = permute(Uinit3DR,[3 2 1]);   
    Uinit3DR = flipdim(Uinit3DR,3); 
    %Reset thetax:    
    thetax = 0;  
    end
    if(thetay==pi/2)  %y-90 transpose
    %Do a simple transpose for speed:
    Uinit3DR = permute(Uinit3DR,[1 3 2]);      
    Uinit3DR = flipdim(Uinit3DR,2); 
    %Reset thetay:    
    thetay = 0;  
    end
    if(thetaz==pi/2)  %z-90 transpose
    %Do a simple transpose for speed:
    Uinit3DR = permute(Uinit3DR,[2 1 3]); 
    Uinit3DR = flipdim(Uinit3DR,1); 
    %Reset thetaz:    
    thetaz = 0;        
    end
end

if((thetax==0 && thetay==0) && thetaz==0)
    Uinit3D = Uinit3DR;
else
  fprintf('VRB:  Rotating vortex ring through nearest neighbor interpolation...')
  for i=1:sizeu(1)
    for j=1:sizeu(2)
        for k=1:sizeu(3)
            %Get original position vector:
            xi = xmin+(j-1)*h;
            yi = ymin+(i-1)*h;
            zi = zmin+(k-1)*h;
            %Rotate vector:
            xyziR = (Rx*Ry*Rz)*[xi;yi;zi];
            xiR = xyziR(1);
            yiR = xyziR(2);
            ziR = xyziR(3);            
            %Compute indices of rotated vector
            ir = ceil((yiR-yminR)/h + 1);
            if(ir>sizeuR(1))
                ir = sizeuR(1);
            end     
            jr = ceil((xiR-xminR)/h + 1);
            if(jr>sizeuR(2))
                jr = sizeuR(2);
            end               
            kr = ceil((ziR-zminR)/h + 1);
            if(kr>sizeuR(3))
                kr = sizeuR(3);
            end
            %Copy value:
            Uinit3D(i,j,k) = Uinit3DR(ir,jr,kr);
        end
    end
  end
fprintf('Done!\n');
end

%Now add new VR to U:
if(sign(s)==-1)
    %Normalize old and new U:
    ampU    = max(abs(U(:)));
    ampUnew = max(abs(Uinit3D(:)));
    %Multiply new solution into U:
    U = (U./ampU).*(Uinit3D./ampUnew);
    %Rescale back to new U amp
    U = ampUnew.*U;
elseif(sign(s)==1)
    %Add new vortex to grid:
    U = U + Uinit3D;
end
clear R;
clear Theta;
clear Finterpv;
clear Uinit3D;
clear XR;
clear YR;
clear ZR;
clear Uinit3DR;
if(plot_profiles==2) %close figures:
    close(f1d2); close(f2d1); close(f2d1phase); close(f2d2); close(f2d2phase); close(f2d3); close(f2d4);
end
disp('Done!');
return
end

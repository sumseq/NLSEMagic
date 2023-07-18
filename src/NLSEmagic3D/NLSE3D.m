%NLSE3D  Full Research Driver Script
%Program to integrate the three-dimensional Nonlinear Shrodinger Equation:
%i*Ut + a*(Uxx+Uyy+Uzz) - V(x,y,z)*U + s*|U|^2*U = 0.
%
%2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

%Clear any previous run variables and close all figures:
munlock; close all; clear all; clear classes; clear functions; pause(0.5);
format long;
twall = tic; %Used to calculate total wall time.
%------------------Simulation parameters----------------------
endtw          = 50;     %Desireed end time fo the simulation - may be slightly altered.
chunksizev     = [0];     %Run CUDA codes with different chunksizes.
numframesv     = [50];    %Number of times to view/plot/analyze solution.
hv             = [1/1.5];%Spatial step-size.
k              = 0;      %Time step-size.  Set to 0 to auto-compute largest stable size.
ICv            = [3];    %Select intial condition (1: linear guassian, 2: bright vortex rings, 3: dark vortex rings).
methodv        = [1];    %Spatial finite-difference scheme: 1:CD O(h^2), 2:2SHOC O(h^4), 11:Script CD, 12:Script 2SHOC.
runs           = 1;      %Number of times to run simulation.
xreswantedv    = [0];    %Desired grid size (N, N x N x N) (Overwrites IC auto grid-size calculations, and may be slightly altered)
dv             = [6];    %Vortex ring radius.
Av             = [0];    %Amplitude of azimuthal mode 2 perturbation of VR.
qv             = [0];    %Vortex ring offset for scattering and co-planer studies.
rmaxv          = [0];    %Set x/y grid size to be rmaxv(ri) from edge of VR (MSD tests).
cudav          = [0];    %Use CUDA integrators (0: no, 1: yes) 
precisionv     = [1];    %Use single (1) or double (2) precision integrators.
BCv            = [0];    %Overides boundary condition choice of IC (0:use IC BC 1:Dirichlet 2:MSD 3:Lap0.
tol            = 20;     %Modulus-squared tolerance to detect blowup.
%------------------Analysis switches----------------------
calc_mass      = 0;      %Calculates change in mass.
add_pert       = 0.0;    %Add uniform random perturbation of size e=add_pert.
track_ring     = 0;      %Track vortex ring along x=0 2D cut.
%------------------Simulation switches--------------------
pause_init     = 1;      %Pause initial condition before integation.
exit_on_end    = 0;      %Close MATLAB after simulation ends.
%------------------Plotting switches----------------------
show_waitbar   = 1;      %Show waitbar with completion time estimate.
plot_3D_vol    = 1;      %Plot 3D volumetric rendering (if plot tags below set).
smooth_3d      = 0;      %Smooth solution for volumetric renders.
high_qual      = 0;      %1: Use high-quality volumetric rending (slower). 0: don't.
plot_2D_cuts   = 0;      %Plot 2D cuts of solution along x=0,y=0,z=0 (if plot tags below set).
plot_modulus2  = 1;      %Plot modulus-squared of solution (2D cuts and/or volumeric render).
plot_vorticity = 0;      %Plot vorticity of solution (2D cuts and/or volumeric render).
vorticity_eps  = 0.001;  %Magnitude of added eps of denominator of vorticity (to avoid singularity).
plot_phase     = 0;      %Plot phase (arg(Psi)) of solution (2D cuts only).
%------------------Output switches----------------------
save_movies    = 0;      %Generate a gif (1) or avi (2) movie of simulation.
save_images    = 0;      %Saves images of each frame into eps (1) or jpg (2). 
disp_pt_data   = 0;      %Output value of Psi(5,5) at each frame.
save_plots     = 0;      %Save result figures into jpg, eps, and fig.
%---------------------------------------------------------

%Set filenames for CUDA timing run:
if(((length(cudav)>1)||length(methodv)==4) && length(xreswantedv)>1)
  str = computer;
  if(strcmp(str, 'PCWIN'))   
      strcuda = 'GT430';
  elseif(strcmp(str, 'PCWIN64'))     
      strcuda = 'GTX580';
  elseif(strcmp(str, 'GLNX86'))
      strcuda = 'GTX260';
  else
      strcuda = 'UNKNOWN';
  end  
  cudafn1 =  [strcuda,'_3D_RES', num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),...
              '_T',num2str(ceil(endtw/k)),'_Fr',num2str(max(numframesv))];          
  diary([cudafn1,'.txt']);
end

%Start parameter loops - for single run, all vectors are of length 1.
for cuda=cudav
for precision = precisionv
for IC=ICv   
for method=methodv
if(((length(cudav)>1)||length(methodv)==4) && length(xreswantedv)>1) 
  cudafn =  [cudafn1,'_IC',num2str(IC),'_M',num2str(method),'_P',...
                 num2str(precision),'_CUDA',num2str(cuda),'.txt'];    
  fid = fopen(cudafn,'wt');  
end    
for xreswanted=xreswantedv

%Set variables for co-planer merge tests:
if(length(qv)>1)    
   mergetime = zeros(size(qv));
   mergez    = mergetime;
end

%VR seperation loop for mergers and scat results:
for qi=1:length(qv)    
    q = qv(qi);
    mergehappened=0;  
    if(length(qv)>1)
        disp(['q: ',num2str(q)]);
    end
    
%VR radius loop:    
for di=1:length(dv)
    d = dv(di);
    if(length(dv)>1)
        disp(['d: ',num2str(d)]);
    end
    %Make copy of rmaxv for MSD tests:
    rmaxvmsd = rmaxv;    
    
for ai=1:length(Av)       
bi = 0;    
for BC_i=BCv
bi = bi+1;    
ni = 0;
numframetimes     = zeros(size(numframesv));
numframechunksize = zeros(size(numframesv));
for numframes = numframesv    
ni = ni+1;  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION PARAMETERS        %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
if(IC==1)   %Linear EXP
    s  = 0;
    a  = 1; 
    BC = 1;
    ymax = sqrt(-2*a*log(sqrt(eps)));
    xmin = -ymax;    xmax =  ymax;    ymin = -ymax;
    zmin = -ymax;    zmax =  ymax;
elseif(IC==2)  %Bright vortex rings  
    s        = 1;
    a        = 1;   
    amp2     = 1;
    BC       = 1;
    %All OMs need to be the same for the same amplitude:
    OM       = s*amp2/3;    
    opt      = 0;
    c        = 0;
    m        = 1;
    %Each vortrings(:,i) places a vortex ring on the grid:
                     %m,OM, x0, y0, z0,thetax,thetay,thetaz,vx, vy, vz, phase,d,opt_2D,ea,eb
    vortrings(:,1) = [m OM  0   0   0  0      0      0      0.0 0.0 0.0 0     d opt,   1, 1];   
elseif(IC==3)   %Dark vortex rings and lines
    s        = -1;
    a        = 1;
    amp2     = 1;           %Modulus-squared background amplitude.
    amp      = sqrt(amp2);  %Amplitude of initial Psi.
    BC       = 2;
    OM       = -1; 
    opt      = 1;    %Optimize vortex ring using 2D PDE
    %d       = 10;   %Overide VR radius.
    %q       = 5;    %Overide VR seperation distance.
    ea       = Av(ai); %Set mode amplitude.
    m        = 1;
    %Set vortex core parameter for VR velocity:
    if(abs(m)==1)
        L0 = 0.380868;    
    elseif(abs(m)==2)
        L0 = 0.133837;
    elseif(abs(m)==3)
        L0 = 0.070755;
    elseif(abs(m)==4)
        L0 = 0.044567;
    elseif(abs(m)==5)
        L0 = 0.030981;
    end
    %Set VR velocity and alter due to amp of pert.
    c   = -((a*m)/d)*(log((8*d)/(m*sqrt(a/(-OM)))) + L0 - 1);      
    c   = c*(1+0.25*(1-(1+ea)));  
    %Set charge 2 velocity for loop tests:
    c2 = -((a*2)/d)*(log((8*d)/(2*sqrt(a/(-OM)))) + 0.133837 - 1);   
    coplane = 0;  %set to 1 for co-plane merge tests.
    
    %Each vortrings(:,i) places a vortex ring on the grid:
    %                 m om  x0  y0  z0  tx  ty tz  vx    vy    vz    phase d opt ea   eb     
    vortrings(:,1) = [m OM  0   0   0   0   0  0   0.0   0.0   0.0   0     d opt ea+1 1 ];   
      
    %Each vortlines(:,i) places a vortex line on the grid:
    %                  m om  x0 y0  z0 vx  vy   vz  phase     
    %vortlines(:,1) = [1 OM  0  0   0  0.0 0.0  0.0 0    ];
end

%Begin rmax loop for MSD tests:
if(BC==2)
    rmaxv_msd = rmaxv;
end
for rmaxi=1:length(rmaxv)   
     
%Now set up vortex IC minimum grid sizes:    
if(IC>1) 
    ymax=0; ymin=0; xmax=0; xmin=0; zmax=0; zmin=0;
    if(exist('vortrings','var'))        
       num_vort = length(vortrings(1,:));        
    for vi=1:num_vort
        if(IC==2)
            B  = sqrt((3*vortrings(2,vi))/s);
            C  = sqrt((3*vortrings(2,vi))/a);
            R0 = sqrt((2*vortrings(1,vi)^2*a)/vortrings(2,vi));
            if(rmaxv(1)==0)              
               rmax  = R0 + 20;%(1/C)*asech(sqrt(eps)/B) + R0;
            else
               rmax = R0 + rmaxv(rmaxi);
            end            
        elseif(IC==3)
            m  = vortrings(1,vi);
            OM = vortrings(2,vi);
            reps = 0.006*(OM/s);
            if(rmaxv(1)==0)
               %rmax = sqrt( -(a*m^2)/(s*reps));                 
               rmax = 20 + abs(m)*sqrt(-a/OM) + max(Av)*d;%*sqrt(a/-OM);
            else
               rmax = rmaxv(rmaxi);
            end
        end        
        if(vi==1)
           xmax = vortrings(3,vi) + rmax + vortrings(13,vi);
           ymax = vortrings(4,vi) + rmax + vortrings(13,vi);
           zmax = vortrings(5,vi) + rmax;        
           xmin = vortrings(3,vi) - rmax - vortrings(13,vi);
           ymin = vortrings(4,vi) - rmax - vortrings(13,vi);
           zmin = vortrings(5,vi) - rmax;            
        end                  
        xmaxt = vortrings(3,vi) + rmax + vortrings(13,vi);
        ymaxt = vortrings(4,vi) + rmax + vortrings(13,vi);
        zmaxt = vortrings(5,vi) + rmax;        
        xmint = vortrings(3,vi) - rmax - vortrings(13,vi);
        ymint = vortrings(4,vi) - rmax - vortrings(13,vi);
        zmint = vortrings(5,vi) - rmax;
        
        if(zmaxt>=zmax), zmax = zmaxt; end
        if(xmaxt>=xmax), xmax = xmaxt; end
        if(ymaxt>=ymax), ymax = ymaxt; end
        if(zmint<=zmin), zmin = zmint; end
        if(xmint<=xmin), xmin = xmint; end   
        if(ymint<=ymin), ymin = ymint; end   
    end
    
    end %vortring
    
    %Manual overide of domain sizes:
    %Elongate domain for single/loop ring travel
    zmin = zmin + c*endtw;    
end %IC>1
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('----------------------------------------------------');
disp('--------------------NLSE3D--------------------------');
disp('----------------------------------------------------');
disp('Checking parameters...');

%Check if multiple BCs are being used:
if(BCv(1)~=0)
   BC = BC_i;
end

%Step size loop (for scheme order analysis)
hi=0;
for h = hv  
hi = hi+1;

%Overide limits for CUDA timings:
if(xreswanted>0)
    xmax =  (xreswanted-1)*h/2;    xmin = -(xreswanted-1)*h/2;
    ymax =  (xreswanted-1)*h/2;    ymin = -(xreswanted-1)*h/2;
    zmax =  (xreswanted-1)*h/2;    zmin = -(xreswanted-1)*h/2;
end

%Adjust ranges so they are in exact units of h:
xmin = xmin - rem(xmin,h); ymin = ymin - rem(ymin,h); zmin = zmin - rem(zmin,h);
xmax = xmax - rem(xmax,h); ymax = ymax - rem(ymax,h); zmax = zmax - rem(zmax,h);

%Update rmaxv accordingly
rmaxv(rmaxi) = rmaxv(rmaxi)-max(abs([rem(xmin,h), rem(xmax,h),...
                                     rem(ymin,h), rem(ymax,h),...
                                     rem(zmin,h), rem(zmax,h)]));
%MSD test, make custom rmaxv for MSD:
if(BC==2)
    rmaxv_msd(rmaxi) = rmaxv(rmaxi);
end
%Set up grid:    
xvec  = xmin:h:xmax;  xres  = length(xvec);
yvec  = ymin:h:ymax;  yres  = length(yvec);
zvec  = zmin:h:zmax;  zres  = length(zvec);
%Need meshgrid matrices to make initial conditions:
[X,Y,Z] = meshgrid(xvec,yvec,zvec);
[M,N,L] = size(X);

%Set up CUDA info
if(cuda==1)    
    cudachk = 0;
    str = computer;
    if( strcmp(str, 'PCWIN') || strcmp(str, 'PCWIN64'))    
        if(exist([matlabroot '/bin/nvmex.pl'],'file')~=0)
            cudachk=1;
        end
    else
        if(exist('/dev/nvidia0','file')~=0)
            cudachk=1;
        end
    end
    if(cudachk==1)
        comp_cap = getCudaInfo;
        %Check precision compatability:
        if(precision==2 && (comp_cap(1)==1 && comp_cap(2)<3))
           precision = 1;
           disp('WARNING:  Precision changed to SINGLE because your CUDA card does not support double precision.');
        end      
        if(precision==1 && method==1)            
            cudablocksizex = 16;            
            cudablocksizey = 16;
            cudablocksizez = 4;            
        elseif(precision==1 && method==2)            
            cudablocksizex = 12;            
            cudablocksizey = 12;
            cudablocksizez = 4;   
        elseif(precision==2 && method==1)            
            cudablocksizex = 10;           
            cudablocksizey = 10;    
            cudablocksizez = 5;
        elseif(precision==2 && method==2) 
            cudablocksizex = 8;           
            cudablocksizey = 8;    
            cudablocksizez = 6; 
        end   
        
        sharedmemperblock = (cudablocksizex+2)*(cudablocksizey+2)*(cudablocksizez+2)*(3+2*method)*(4*precision)/1000;
        numcudablocksx    = ceil(M/cudablocksizex);    
        numcudablocksy    = ceil(N/cudablocksizey);
        numcudablocksz    = ceil(L/cudablocksizez);
        numcudablocks     = numcudablocksx*numcudablocksy*numcudablocksz;
        
        %For MSD BC, need to check this:
        if(BC==2)
            msdalterflag=0;
            if(M - cudablocksizex*(numcudablocksx-1) == 1)
                disp('MSD CUDA ERROR: M (yres) is one cell greater than CUDA block x-direction,')
                ymax = ymax-h;   
                ymin = ymin+h;   
                msdalterflag=1;
                disp(['adjusting ymax to ',num2str(ymax),' and ymin to ',num2str(ymin),' to compensate,']);                
            end
            if(N - cudablocksizey*(numcudablocksy-1) == 1) 
                disp('MSD CUDA ERROR: N (xres) is one cell greater than CUDA block y-direction')
                xmax = xmax-h;     
                xmin = xmin+h;
                msdalterflag=1;
                disp(['adjusting xmax to ',num2str(xmax),' and xmin to ',num2str(xmin),' to compensate,']);                   
            end    
            if(L - cudablocksizez*(numcudablocksz-1) == 1) 
                disp('MSD CUDA ERROR: L (zres) is one cell greater than CUDA block z-direction')
                zmax = zmax-h;           
                zmin = zmin+h;
                msdalterflag=1;
                disp(['adjusting zmax to ',num2str(zmax),' and zmin to ',num2str(zmin),' to compensate,']);                  
            end            
            
            if(msdalterflag==1)
               %Have to update rmax vector for msd tests for accurate results
               rmaxv_msd(rmaxi) = rmaxv(rmaxi)-h;
            end
            
            %Now, recompute grid:
            xvec  = xmin:h:xmax;  xres  = length(xvec);
            yvec  = ymin:h:ymax;  yres  = length(yvec);
            zvec  = zmin:h:zmax;  zres  = length(zvec);            
            [X,Y,Z] = meshgrid(xvec,yvec,zvec);
            
            %Reset numblocks with new gridsize:
            [M,N,L] = size(X);
            numcudablocksx = ceil(M/cudablocksizex);    
            numcudablocksy = ceil(N/cudablocksizey);
            numcudablocksz = ceil(L/cudablocksizez);
            numcudablocks  = numcudablocksx*numcudablocksy*numcudablocksz;
        end       
    else
        disp('Sorry, it seems CUDA is not installed');    
        cuda=0;        
    end
end%cuda1


if(length(rmaxv)>1)
   disp(['Rpad:  ',num2str(rmaxv(rmaxi))]);
end

for run_i = 1:runs %Run loop (for averaging timings)

%Initialize solutiona and potential matrices:
U     = zeros(size(X));
V     = zeros(size(X)); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION FORMULATION       %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
disp('----------------------------------------------------');
disp('Setting up initial condition...---------------------');
disp(['GridSize(x,y,z): ',num2str(xres),'x',num2str(yres),'x',num2str(zres), ' = ',num2str(xres*yres*zres),' = ',num2str(xres*yres*zres*16/1000),'KB']);
%Formulate Initial Condition:
if(IC>1) %Vortex rings and lines:
if(IC==3)
   U = U + sqrt(amp2);
end
if(exist('vortrings','var'))
for vi=1:num_vort
  U = NLSE_add_vortex_ring(vortrings(1,vi),vortrings(2,vi),vortrings(3,vi),...
                           vortrings(4,vi),vortrings(5,vi),vortrings(6,vi),...
                           vortrings(7,vi),vortrings(8,vi),vortrings(9,vi),...
                           vortrings(10,vi),vortrings(11,vi),vortrings(12,vi),...
                           U,h,a,s,X,Y,Z,vortrings(13,vi),vortrings(14,vi),...
                           vortrings(15,vi),vortrings(16,vi));
end
end
if(exist('vortlines','var'))
   num_line = length(vortlines(1,:));
   for vi=1:num_line
      U = NLSE_add_vortex_line(vortlines(1,vi),vortlines(2,vi),vortlines(3,vi),...
                           vortlines(4,vi),vortlines(5,vi),vortlines(6,vi),...
                           vortlines(7,vi),vortlines(8,vi),vortlines(9,vi),...                           
                           U,h,a,s,amp,X,Y,Z);
   end
end
elseif(IC==1) %Linear steady-state exp profile   
    V     = (X.^2 + Y.^2 + Z.^2)/a;
    U     = exp(-(X.^2 + Y.^2 + Z.^2)/(2*a));
    Ureal = U;
end

%Add random perturbation if specified:
if(add_pert > 0)
    U = U.*(1 + add_pert.*rand(size(U)));
end

if(max(real(U(:))) > tol || sum(isnan(U(:))) > 0)
   disp('Initial Condition Has a PROBLEM'); 
   break;
end

%-------------------------------------------------------------------------%
%Compute stability bounds for time step k:
if(k==0)  
   hmin = min(hv);  
   if(method==1 || method==11)  
      G = [12,11,10,9,3,2,1,0]; 
      klin = hmin^2/(3*sqrt(2)*a);
   elseif((method==2 || method==12) || length(methodv)>1)
      G = (1/12)*[192,191,190,189,174,173,172,156,155,138,36,21,20,6,5,4,-9,-10,-11,-12];
      klin = (3/4)*hmin^2/(3*sqrt(2)*a);
   end
    
   Lmaxes = zeros(size(G));
   %Compute Lmax 
   for gi=[1:length(G)]
        Lmaxes(gi) = max(abs((hmin^2/a)*(s*U(:).*conj(U(:)) - V(:)) - G(gi)));           
   end
   %Compute B vector depending on BC
   if(BC==1)
       Bmax = 0;
   elseif(BC==2)           
       ub1   = U(2,2:end-1,2:end-1);     ub2 = U(2:end-1,2,2:end-1);     ub3 = U(2:end-1,2:end-1,2);
       ub4   = U(end-1,2:end-1,2:end-1); ub5 = U(2:end-1,end-1,2:end-1); ub6 = U(2:end-1,2:end-1,end-1); 
       UB_1  = [ub1(:); ub2(:); ub3(:); ub4(:); ub5(:); ub6(:)];
       Ut    = NLSE3D_F(U,V,hmin,s,a,method,BC,(1/hmin^2),(a/12),(a/(6*hmin^2)),(1/a));
       utb1  = Ut(2,2:end-1,2:end-1);     utb2 = Ut(2:end-1,2,2:end-1);     utb3 = Ut(2:end-1,2:end-1,2);
       utb4  = Ut(end-1,2:end-1,2:end-1); utb5 = Ut(2:end-1,end-1,2:end-1); utb6 = Ut(2:end-1,2:end-1,end-1); 
       UtB_1 = [utb1(:); utb2(:); utb3(:); utb4(:); utb5(:); utb6(:)];                     
       Bmax = real(max((hmin^2/(1i*a))*UtB_1./UB_1));
       clear UtB_1 UB_1 Ut ub1 ub2 ub3 ub4 ub5 ub6 utb1 utb2 utb3 utb4 utb5 utb6;
    elseif(BC==3)
       u1 = U(1,:,:);   u2 = U(:,1,:);   u3 = U(:,:,1);
       u4 = U(end,:,:); u5 = U(:,end,:); u6 = U(:,:,end);
       UB = [u1(:); u2(:); u3(:); u4(:); u5(:); u6(:)];           
       v1 = V(1,:,:);   v2 = V(:,1,:);   v3 = V(:,:,1);
       v4 = V(end,:,:); v5 = V(:,end,:); v6 = V(:,:,end);
       VB = [v1(:); v2(:); v3(:); v4(:); v5(:); v6(:)];  
       Bmax = max((hmin^2/a)*(s*UB.*conj(UB) - VB));
       clear VB UB u1 u2 u3 u4 u5 u6 v1 v2 v3 v4 v5 v6;
   end    
   %Now compute full k bound:
   kfull = (hmin^2/a)*sqrt(8)/max(abs(Bmax),max(abs(Lmaxes)));
   disp(['kmax (linear): ',num2str(klin)]);
   disp(['kmax   (full): ',num2str(kfull)]);
   if(kfull<klin)
      k = 0.6*kfull;
   else
      k = 0.6*klin;
   end 
   disp(['k      (used): ',num2str(k)]);
end

%Compute total number of time steps for simulation:
endt  = endtw - mod(endtw,k); %<--Make endtime multiple of k
if(mod(endtw,k)~=0), disp(['NOTE! End time adjusted to ',num2str(endt)]); end;
steps = floor(endt/k);             %<--Compute number of steps required
if(numframes>=steps), numframes=steps; end;

%Compute number of time steps to compute in mex file:
chunk_size   = ceil(steps/numframes);
extra_steps  = chunk_size*numframes - steps;
if(extra_steps>chunk_size)
    numframes    = ceil(steps/chunk_size);
    disp(['To avoid too much over-time compuations, numframes has been altered to ',num2str(numframes)]);
    extra_steps  = chunk_size*numframes - steps;
end
if(extra_steps>0)
    disp(['NOTE! There will be ',num2str(extra_steps),' extra time steps taken in simulation.']);
    endt = chunk_size*numframes*k;
    steps  = steps + extra_steps;
    disp(['Therefore, true end time will be: ',num2str(endt),' taking ',num2str(steps),' time steps']);
end;

%If first run, print out simulation info:
if(hi==1 && run_i==1)
      
disp('----------------------------------------------------');
disp('3D NLSE Parameters:')
disp(['a:          ',num2str(a)]);
disp(['s:          ',num2str(s)]);
disp(['xmin:       ',num2str(xmin)]);
disp(['xmax:       ',num2str(xmax)]);
disp(['ymin:       ',num2str(ymin)]);
disp(['ymax:       ',num2str(ymax)]);
disp(['zmin:       ',num2str(zmin)]);
disp(['zmax:       ',num2str(zmax)]);
disp(['Endtime:    ',num2str(endt)]);
disp('Numerical Parameters:');
disp(['h:               ',num2str(h)]);
disp(['k:               ',num2str(k)]);
disp(['GridSize(x,y,z): ',num2str(xres),'x',num2str(yres),'x',num2str(zres), ' = ',num2str(xres*yres*zres),' = ',num2str(xres*yres*zres*16/1000),'KB']);
disp(['TimeSteps:       ',num2str(steps)]);
disp(['ChunkSize:       ',num2str(chunk_size)]); 
disp(['NumFrames:       ',num2str(numframes)]);
if(precision==1)
    disp('Precision:  Single');
else
    disp('Precision:  Double');
end
if(method==1)
    disp('Method:     RK4+CD     Order(4,2)');
elseif(method==2)
    disp('Method:     RK4+2SHOC  Order(4,4)');
elseif(method==11)
    disp('Method:     RK4+CD     Order(4,2) SCRIPT');    
elseif(method==12)
    disp('Method:     RK4+2SHOC  Order(4,4) SCRIPT');
end
if(BC==1)
    disp('Boundary Conditions:  Dirichlet');
elseif(BC==2)
    disp('Boundary Conditions:  Mod-Squared Dirichlet');
elseif(BC==3)
    disp('Boundary Conditions:  Uxx+Uyy+Uzz = 0');
end
if(cuda == 1)
    disp( 'CUDA Parameters and Info:');          
    disp(['BlockSize:     ',num2str(cudablocksizex),'x',num2str(cudablocksizey),'x',num2str(cudablocksizez)]);
    disp(['CUDAGridSize:  ',num2str(numcudablocksx),'x',num2str(numcudablocksy*numcudablocksz)]);
    disp(['NumBlocks:     ',num2str(numcudablocks)]); 
    disp(['Shared Memory/Block: ',num2str(sharedmemperblock),'KB']);
    totmemreq = (yres*xres*zres*(9 + 2*(method-1))*(4*precision))/1024;
    disp(['TotalGPUMemReq:      ',num2str(totmemreq), ' KB']); 
end
if(save_movies>1)
    disp('Warning!  Saving movie...');
end
if(save_images==1)
    disp('Warning!  Saving images...');    
end
end %hi==1
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(hi>1)
    disp(['h:          ',num2str(h)]);
    disp(['k:          ',num2str(k)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             PLOTTING SETUP                      %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%

maxMod2U  = max(U(:).*conj(U(:)));
maxRealU  = max(real(U(:)));
fsize     = 18;  %Set font size for figures.
fig_count = 1;

if(plot_vorticity==1 || track_ring==1)        
     [w_x,w_y,w_z,W] = rmc_vorticity3D(U,1/(h*2),vorticity_eps);
end

if(plot_3D_vol==1) %3D Volumetric plots

      xticklabels    = floor( (xmin + 1 +(xmax-1-xmin).*[0 1/4 1/2 3/4 1]));
      yticklabels    = floor( (ymin + 1 +(ymax-1-ymin).*[0 1/4 1/2 3/4 1]));
      zticklabels    = floor( (zmin + 1 +(zmax-1-zmin).*[0 1/4 1/2 3/4 1]));  
      xticklocations = (xticklabels - xmin)./h;
      yticklocations = (yticklabels - ymin)./h;
      zticklocations = (zticklabels - zmin)./h;
        
   if(plot_modulus2==1)    
      fig_cube_mod2  = figure(fig_count);
      fig_count      = fig_count+1;
      set(fig_cube_mod2, 'Name','3D NLSE MOD2 t = 0','NumberTitle','off',...
              'InvertHardcopy','off','Color','k');
      set(fig_cube_mod2, 'PaperPositionMode', 'auto');
      %Plot mod-squared initial condition:
      if(smooth_3d==0)
        % U = squeeze(U(:,floor(length(xvec)/2):end,:));
         plot_cube_mod2 = vol3d('cdata',U.*conj(U),'texture','3D');
      else
         plot_cube_mod2 = vol3d('cdata',smooth3(U.*conj(U)),'texture','3D');  
      end
      colormap(hsv(512));      
      xlim([0 xres]); ylim([0 yres]); zlim([0 zres]);      
      set(gca,'FontSize',fsize);
      set(gca,'XTick',xticklocations,'XTickLabel',xticklabels);
      set(gca,'YTick',yticklocations,'YTickLabel',yticklabels);
      set(gca,'ZTick',zticklocations,'ZTickLabel',zticklabels);
      set(gca,'DataAspectRatio',[xres/abs(xmax-xmin) yres/abs(ymax-ymin) zres/abs(zmax-zmin)]); 
      grid on;
      xlabel('x','FontSize',fsize); ylabel('y','FontSize',fsize); zlabel('z','FontSize',fsize);
      if(save_images==0)
          set(gca,'Color','k','XColor','w','YColor','w','ZColor','w');
      else
          set(gca,'Color','k','XColor','w','YColor','w','ZColor','w');          
      end
      axis vis3d;
      zoom(0.75);
      view([-20 20]);      
      cmax_init = max(U(:).*conj(U(:)));
      if(IC~=2)
         caxis([0 cmax_init]);
      end
      if(save_images==0 && save_movies==0)
         cb = colorbar;
         set(cb,'FontSize',fsize);
      end
      %plot_label = text((1/4)*xres,(1/4)*yres,(3/4)*zres,'Time:      0.00','BackgroundColor','white');

      %Set alphamap:
      if(s<0)
         avec = (0:0.1:log(1000));
         amap = exp(avec)./1000;
         amap = amap(end:-1:1);    
         alphamap(amap);
         alim([0 cmax_init]);    
      else
         avec = [0:0.1:log(1000)];
         amap = exp(avec)./1000;
         alphamap(amap); 
         if(IC==1)
            alphamap('rampup'); 
         end
        % alim([0 cmax_init]);  
      end
   end %Plot_mod2

   if(plot_vorticity==1)  
        fig_cube_vort  = figure(fig_count);
        fig_count = fig_count+1;
        if(smooth_3d==0)
            plot_cube_vort = vol3d('cdata',(W),'texture','3D');
        else
            plot_cube_vort = vol3d('cdata',smooth3(W),'texture','3D');
        end
        set(fig_cube_vort, 'Name','3D NLSE  Vorticity t = 0','NumberTitle','off',...
              'InvertHardcopy','off','Color','k');    
        set(fig_cube_vort, 'PaperPositionMode', 'auto');
        colormap(jet(512));
        xlim([0 xres]); ylim([0 yres]); zlim([0 zres]);       
        set(gca,'FontSize',fsize);
        set(gca,'XTick',xticklocations,'XTickLabel',xticklabels);
        set(gca,'YTick',yticklocations,'YTickLabel',yticklabels);
        set(gca,'ZTick',zticklocations,'ZTickLabel',zticklabels); 
        set(gca,'DataAspectRatio',[xres/abs(xmax-xmin) yres/abs(ymax-ymin) zres/abs(zmax-zmin)]); 
        grid on;
        xlabel('x','FontSize',fsize); ylabel('y','FontSize',fsize); zlabel('z','FontSize',fsize);
        if(save_images==0)
          set(gca,'Color','k','XColor','w','YColor','w','ZColor','w');
        else
          set(gca,'Color','k','XColor','w','YColor','w','ZColor','w');          
        end
        axis vis3d;
        zoom(0.75);
        view([-20 20]);       
        cmax_init = max((W(:)));
        cmin_init = min((W(:)));
        caxis([cmin_init cmax_init]);
        %caxis([-1 1]);
        if(save_images==0 && save_movies==0)
           colorbar;
        end
        alphamap('rampup');       
        %alim([-1 1]);   
   end%plot_vort
end %plot_3D_vol

if(plot_2D_cuts==1)

    if(plot_modulus2==1)
        fig_cuts_mod2  = figure(fig_count);        fig_count = fig_count+1;
        set(fig_cuts_mod2, 'Name','3D NLSE  Mod2 t = 0','NumberTitle','off',...
                  'InvertHardcopy','off','Color','w');
        set(fig_cuts_mod2, 'PaperPositionMode', 'auto');
        axtmp = subplot(1,3,1);
        plot_cuts_mod2_XY = pcolor(squeeze(X(:,:,floor(zres/2))),squeeze(Y(:,:,floor(zres/2))),squeeze(U(:,:,floor(zres/2)).*conj(U(:,:,floor(zres/2)))));
        shading interp;        view([0 90]);        colormap(jet(512));
        axis equal
        xlim([xmin xmax]);        ylim([ymin ymax]);
        %set(axtmp,'DataAspectRatio',[xres/abs(xmax-xmin) yres/abs(ymax-ymin) 1]);
        xlabel('x','FontSize',fsize);        ylabel('y','FontSize',fsize);
        if(s<0)
           cmax_init = max(U(:).*conj(U(:)));
           caxis([0 cmax_init]);
        end
       % time_txt   = ['Time:      0.00'];
       % plot_label = text(0,(3/4)*ymax,0.1,time_txt,'BackgroundColor','white');
        
        axtmp =subplot(1,3,2);
        plot_cuts_mod2_XZ = pcolor(squeeze(X(floor(yres/2),:,:)),squeeze(Z(floor(yres/2),:,:)),squeeze(U(floor(yres/2),:,:).*conj(U(floor(yres/2),:,:))));
        shading interp;        view([0 90]);        colormap(jet(512));
        axis equal;
        xlim([xmin xmax]);        ylim([zmin zmax]);
        %set(axtmp,'DataAspectRatio',[xres/abs(xmax-xmin) zres/abs(zmax-zmin) 1]);
        xlabel('x','FontSize',fsize);        ylabel('z','FontSize',fsize);
        if(s<0)
           cmax_init = max(U(:).*conj(U(:)));
           caxis([0 cmax_init]);        
        end
        
        axtmp =subplot(1,3,3);
        plot_cuts_mod2_YZ = pcolor(squeeze(Y(:,floor(xres/2),:)),squeeze(Z(:,floor(xres/2),:)),squeeze(U(:,floor(xres/2),:).*conj(U(:,floor(xres/2),:))));
        shading interp;        view([0 90]);        colormap(jet(512));
        axis equal;
        xlim([ymin ymax]);        ylim([zmin zmax]);
        %set(axtmp,'DataAspectRatio',[yres/abs(ymax-ymin) zres/abs(zmax-zmin) 1]);
        xlabel('y','FontSize',fsize);        ylabel('z','FontSize',fsize);  
        
    end%plotmod2
   if(plot_phase==1)
        fig_cuts_phase  = figure(fig_count);        fig_count = fig_count+1;
        set(fig_cuts_phase, 'Name','3D NLSE  Phase t = 0','NumberTitle','off',...
                  'InvertHardcopy','off','Color','w');
        set(fig_cuts_phase, 'PaperPositionMode', 'auto');
        axtmp = subplot(1,3,1);
        plot_cuts_phase_XY = pcolor(squeeze(X(:,:,floor(zres/2))),squeeze(Y(:,:,floor(zres/2))),squeeze(angle(U(:,:,floor(zres/2)))));
        shading interp;        view([0 90]);        colormap(jet(512));
        xlim([xmin xmax]);        ylim([ymin ymax]);
        set(axtmp,'DataAspectRatio',[xres/abs(xmax-xmin) yres/abs(ymax-ymin) 1]);
        xlabel('x','FontSize',fsize);        ylabel('y','FontSize',fsize);

        axtmp =subplot(1,3,2);
        plot_cuts_phase_XZ = pcolor(squeeze(X(floor(yres/2),:,:)),squeeze(Z(floor(yres/2),:,:)),squeeze(angle(U(floor(yres/2),:,:))));
        shading interp;        view([0 90]);        colormap(jet(512));
        xlim([xmin xmax]);        ylim([zmin zmax]);
        set(axtmp,'DataAspectRatio',[xres/abs(xmax-xmin) zres/abs(zmax-zmin) 1]);
        xlabel('x','FontSize',fsize);        ylabel('z','FontSize',fsize);
        

        axtmp =subplot(1,3,3);
        plot_cuts_phase_YZ = pcolor(squeeze(Y(:,floor(xres/2),:)),squeeze(Z(:,floor(xres/2),:)),squeeze(angle(U(:,floor(xres/2),:))));
        shading interp;        view([0 90]);        colormap(jet(512));
        xlim([ymin ymax]);        ylim([zmin zmax]);
        set(axtmp,'DataAspectRatio',[yres/abs(ymax-ymin) zres/abs(zmax-zmin) 1]);
        xlabel('y','FontSize',fsize);        ylabel('z','FontSize',fsize);  
    end%plotphase
    if(plot_vorticity==1)
        fig_cuts_vort  = figure(fig_count);        fig_count = fig_count+1;
        set(fig_cuts_vort, 'Name','3D NLSE  Vorticity t = 0','NumberTitle','off',...
                  'InvertHardcopy','off','Color','w');
        set(fig_cuts_vort, 'PaperPositionMode', 'auto');
        axtmp = subplot(1,3,1);
        plot_cuts_vort_XY = pcolor(squeeze(X(:,:,floor(zres/2))),squeeze(Y(:,:,floor(zres/2))),squeeze((W(:,:,floor(zres/2)))));
        shading interp;        view([0 90]);        colormap(jet(512));
        xlim([xmin xmax]);        ylim([ymin ymax]);
        set(axtmp,'DataAspectRatio',[xres/abs(xmax-xmin) yres/abs(ymax-ymin) 1]);
        xlabel('x','FontSize',fsize);        ylabel('y','FontSize',fsize);

        axtmp =subplot(1,3,2);
        plot_cuts_vort_XZ = pcolor(squeeze(X(floor(yres/2),:,:)),squeeze(Z(floor(yres/2),:,:)),squeeze((W(floor(yres/2),:,:))));
        shading interp;        view([0 90]);        colormap(jet(512));
        xlim([xmin xmax]);        ylim([zmin zmax]);
        set(axtmp,'DataAspectRatio',[xres/abs(xmax-xmin) zres/abs(zmax-zmin) 1]);
        xlabel('x','FontSize',fsize);        ylabel('z','FontSize',fsize);
        caxis([0 max(max(abs(squeeze((W(floor(yres/2),:,:))))))]); 
        
        axtmp =subplot(1,3,3);
        plot_cuts_vort_YZ = pcolor(squeeze(Y(:,floor(xres/2),:)),squeeze(Z(:,floor(xres/2),:)),squeeze((W(:,floor(xres/2),:))));
        shading interp;        view([0 90]);        colormap(jet(512));
        xlim([ymin ymax]);        ylim([zmin zmax]);
        set(axtmp,'DataAspectRatio',[yres/abs(ymax-ymin) zres/abs(zmax-zmin) 1]);
        xlabel('y','FontSize',fsize);        ylabel('z','FontSize',fsize);  
        
   end%plotvort

end %plot_2D_cuts
drawnow;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(save_movies>=1 || save_images==1)
if(IC==1)
    fn = 'linear';
elseif(IC==2 || IC==3)
    %Define filenames for movie/image
     sizetmp = size(vortrings);
     fn = ['VR','_d',num2str(d),'_m',num2str(m),'_q',num2str(q),'_a',num2str(a),'_s',num2str(s),'_OM',num2str(OM),...
               '_T',num2str(endt),'_M',num2str(method),'_P',num2str(precision),'_BC',num2str(BC),'_h',num2str(h),'_k',num2str(k)];
     fn = strrep(fn,'.','o');     
end
end

if(show_waitbar==1)
    fig_waitbar = waitbar(0,'Estimated time to completion ???:??');
    set(fig_waitbar, 'Name','Simulation 0% Complete','NumberTitle','off');
    waitbar(0,fig_waitbar,'Estimated time to completion ???:??');  
end

if(pause_init==1)
    disp('Initial condition displayed.  Press a key to start simulation');
    pause;
end

if(save_movies==1 || save_movies==3)
    if(plot_3D_vol==1)
       if(plot_modulus2==1)
          gifname = [fn,'_MOD2_3DVOL.gif'];
          I = getframe(fig_cube_mod2);
          I = frame2im(I);
          [XX, map] = rgb2ind(I, 128);
          imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
          'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
       end
       if(plot_vorticity==1)
          gifname = [fn,'_VORT_3DVOL.gif'];
          I = getframe(fig_cube_vort);
          I = frame2im(I);
          [XX, map] = rgb2ind(I, 128);
          imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
          'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
       end       
    end
    if(plot_2D_cuts==1)
        if(plot_modulus2==1)
          gifname = [fn,'_MOD2_2DCUTS.gif'];
          I = getframe(fig_cuts_mod2);
          I = frame2im(I);
          [XX, map] = rgb2ind(I, 128);
          imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
          'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
       end
       if(plot_phase==1)
          gifname = [fn,'_PHASE_2DCUTS.gif'];
          I = getframe(fig_cuts_phase);
          I = frame2im(I);
          [XX, map] = rgb2ind(I, 128);
          imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
          'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
       end 
       if(plot_vorticity==1)
          gifname = [fn,'_VORT_2DCUTS.gif'];
          I = getframe(fig_cuts_vort);
          I = frame2im(I);
          [XX, map] = rgb2ind(I, 128);
          imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
          'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
       end 
    end
end
if(save_movies==2 || save_movies==3)
    comp_opt = 'none';
    if(plot_3D_vol==1)
        if(plot_modulus2==1)
           mov_cube_mod2 = avifile([fn,'_MOD2_3DVOL.avi'],'compression',comp_opt,'quality',100);    
           mov_cube_mod2 = addframe(mov_cube_mod2,getframe(fig_cube_mod2)); 
        end
        if(plot_vorticity==1)
           mov_cube_vort = avifile([fn,'_VORT_3DVOL.avi'],'compression',comp_opt,'quality',100);    
           mov_cube_vort = addframe(mov_cube_vort,getframe(fig_cube_vort)); 
        end
    end
    if(plot_2D_cuts==1)
        if(plot_modulus2==1)
           mov_cuts_mod2 = avifile([fn,'_MOD2_2DCUTS.avi'],'compression',comp_opt,'quality',100);    
           mov_cuts_mod2 = addframe(mov_cuts_mod2,getframe(fig_cuts_mod2)); 
        end
        if(plot_phase==1)
           mov_cuts_phase = avifile([fn,'_PHASE_2DCUTS.avi'],'compression',comp_opt,'quality',100);    
           mov_cuts_phase = addframe(mov_cuts_phase,getframe(fig_cuts_phase)); 
        end
        if(plot_vorticity==1)
           mov_cuts_vort = avifile([fn,'_VORT_2DCUTS.avi'],'compression',comp_opt,'quality',100);    
           mov_cuts_vort = addframe(mov_cuts_vort,getframe(fig_cuts_vort)); 
        end
    end    
end

if(save_images==1)
    comp_opt = 'none';
    if(plot_3D_vol==1)
        if(plot_modulus2==1)
           set(fig_cube_mod2, 'PaperPositionMode', 'auto'); 
           set(fig_cube_mod2,'InvertHardcopy','off');           
           print('-djpeg','-r100','-opengl',[fn, '_T0', '_MOD2_3DVOL'],['-f',num2str(fig_cube_mod2)]);            
        end
        if(plot_vorticity==1)
           set(fig_cube_vort, 'PaperPositionMode', 'auto'); 
           set(fig_cube_vort,'InvertHardcopy','off');
           fignum = ['-f',num2str(fig_cube_vort)];
           print('-djpeg','-r100',[fn, '_T0','_VORT_3DVOL'],fignum);     
        end
    end
    if(plot_2D_cuts==1)
        if(plot_modulus2==1)
           set(fig_cuts_mod2, 'PaperPositionMode', 'auto'); 
           set(fig_cuts_mod2,'InvertHardcopy','off');
           fignum = ['-f',num2str(fig_cuts_mod2)];
           print('-depsc','-r100',[fn, '_T0','_MOD2_2DCUTS'],fignum); 
           print('-djpeg','-r100',[fn, '_T0','_MOD2_2DCUTS'],fignum); 
        end
        if(plot_vorticity==1)
           set(fig_cuts_vort, 'PaperPositionMode', 'auto'); 
           set(fig_cuts_vort,'InvertHardcopy','off');
           fignum = ['-f',num2str(fig_cuts_vort)];
           print('-depsc','-r100',[fn, '_T0','_VORT_2DCUTS'],fignum);   
           print('-djpeg','-r100',[fn, '_T0','_VORT_2DCUTS'],fignum); 
        end
        if(plot_phase==1)
           set(fig_cuts_phase, 'PaperPositionMode', 'auto'); 
           set(fig_cuts_phase,'InvertHardcopy','off');
           fignum = ['-f',num2str(fig_cuts_phase)];
           print('-depsc','-r100',[fn, '_T0','_PHASE_2DCUTS'],fignum); 
           print('-djpeg','-r100',[fn, '_T0','_PHASE_2DCUTS'],fignum); 
        end
     end    
end

%Initialize counters, etc:
calcn  = 0;
n      = 0;
s_count= 1;
ttotal = tic; %Start total time timer
tcomp  = 0;   %Initialize compute-time to 0

%Initialize RK4 matrices:
if(method>10) %Need these matrices for non-mex code:
   Uk_tmp = zeros(size(U));
   k_tot  = zeros(size(U));
end

if(track_ring==1)
    zv1       = zeros(num_vort,numframes);
    yv1       = zeros(num_vort,numframes);
    ziv1      = zeros(num_vort,numframes);
    yiv1      = zeros(num_vort,numframes);
    zv2       = zeros(num_vort,numframes);
    yv2       = zeros(num_vort,numframes);
    ziv2      = zeros(num_vort,numframes);
    yiv2      = zeros(num_vort,numframes);
    v_trackY1 = zeros(num_vort,numframes);
    v_trackZ1 = zeros(num_vort,numframes);  
    v_trackY2 = zeros(num_vort,numframes);
    v_trackZ2 = zeros(num_vort,numframes);
end

plottime  = zeros(numframes,1);
if(IC==1)
errorveci = zeros(numframes,1);
errorvecr = zeros(numframes,1);
errorvecm = zeros(numframes,1);
end

if(save_images==1 && pause_init==1)
   disp('Initial images saved, press enter to start simulation...');
   pause;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             BEGIN SIMULATION                    %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
for chunk_count = 1:numframes
   
   %Start chunk-time timer (includes plots)
   if(show_waitbar==1), tchunk  = tic; end    
   tcompchunk = tic;  %Start compute-time timer
   
   if(method==1)    
    if(precision==1)
        if(cuda==0)
            U = NLSE3D_TAKE_STEPS_CD_F(U,V,s,a,h^2,BC,chunk_size,k);
        else
            U = NLSE3D_TAKE_STEPS_CD_CUDA_F(U,V,s,a,h^2,BC,chunk_size,k);
        end
    elseif(precision==2)
        if(cuda==0)   
            U = NLSE3D_TAKE_STEPS_CD(U,V,s,a,h^2,BC,chunk_size,k);
        else
            U = NLSE3D_TAKE_STEPS_CD_CUDA_D(U,V,s,a,h^2,BC,chunk_size,k);
        end 
    end
   elseif(method==2)      
     if(precision==1)
        if(cuda==0)            
            U = NLSE3D_TAKE_STEPS_2SHOC_F(U,V,s,a,h^2,BC,chunk_size,k);
        else
            U = NLSE3D_TAKE_STEPS_2SHOC_CUDA_F(U,V,s,a,h^2,BC,chunk_size,k);
        end
    elseif(precision==2)
        if(cuda==0)   
            U = NLSE3D_TAKE_STEPS_2SHOC(U,V,s,a,h^2,BC,chunk_size,k);
        else
            U = NLSE3D_TAKE_STEPS_2SHOC_CUDA_D(U,V,s,a,h^2,BC,chunk_size,k);
        end 
     end    
   else  %Old code for trial methods / mex timings    
    %Do divisions first to save compute-time:
    k2    = k/2;          k6    = k/6;    l_h2  = 1/h^2;    a_12  = a/12;
    a_6h2 = a/(6*h^2);    l_a   = 1/a;
    
    for nc = 1:chunk_size        
       %Start Runga-Kutta:     
       k_tot   = NLSE3D_F(U,V,h,s,a,method,BC,l_h2,a_12,a_6h2,l_a); %K1   
       %----------------------------     
       Uk_tmp  = U + k2*k_tot;     
       Uk_tmp  = NLSE3D_F(Uk_tmp,V,h,s,a,method,BC,l_h2,a_12,a_6h2,l_a); %K2
       k_tot   = k_tot + 2*Uk_tmp; %K1 + 2K2
       %----------------------------        
       Uk_tmp  = U + k2*Uk_tmp;     
       Uk_tmp  = NLSE3D_F(Uk_tmp,V,h,s,a,method,BC,l_h2,a_12,a_6h2,l_a); %K3
       k_tot   = k_tot + 2*Uk_tmp; %K1 + 2K2 + 2K3
       %----------------------------
       Uk_tmp  = U + k*Uk_tmp;     
       k_tot   = k_tot + NLSE3D_F(Uk_tmp,V,h,s,a,method,BC,l_h2,a_12,a_6h2,l_a); %K1 + 2K2 + 2K3 + K4
       %-------------------------------       
       U = U + k6*k_tot; %New time step
    end %chunksize
	   clear functions;
   end%method step    
     
   %ADD: Add code here to change chunk_size so simulation does not go over
   %endt (if not doing timings)
   tcomp = tcomp + toc(tcompchunk); %Add comp-chunk time to compute-time
   n = n + chunk_size;   
   
%Detect blow-up:
if(max(abs(U(:)).^2) > tol || sum(isnan(U(:))) > 0)
  disp('CRAAAAAAAAAAAAAAAASSSSSSSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHH');
  disp(['CRASH!  h=',num2str(h),' k=',num2str(k),' t=',num2str(n*k,'%.2f')]);  
  if( sum(isnan(U(:))) > 0)
  disp('NAN Found!');
  else
  disp('Tolerance Exceeded!');
  end  
  disp('CRAAAAAAAAAAAAAAAASSSSSSSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHH');
  break;
end

%On-the-fly stability checker (not used)
%    %Compute Lmax 
%    for gi=[1:length(G)]
%         Lmaxes(gi) = max((h^2/a)*(s*U(:).*conj(U(:)) - V(:)) - G(gi));           
%    end
%    %Compute B vector depending on BC
%    if(BC==1)
%        Bmax = 0;
%    elseif(BC==2)           
%        ub1   = U(2,2:end-1,2:end-1);     ub2 = U(2:end-1,2,2:end-1);     ub3 = U(2:end-1,2:end-1,2);
%        ub4   = U(end-1,2:end-1,2:end-1); ub5 = U(2:end-1,end-1,2:end-1); ub6 = U(2:end-1,2:end-1,end-1); 
%        UB_1  = [ub1(:); ub2(:); ub3(:); ub4(:); ub5(:); ub6(:)];
%        Ut    = NLSE3D_F(U,V,h,s,a,method,BC,(1/h^2),(a/12),(a/(6*h^2)),(1/a));
%        utb1  = Ut(2,2:end-1,2:end-1);     utb2 = Ut(2:end-1,2,2:end-1);     utb3 = Ut(2:end-1,2:end-1,2);
%        utb4  = Ut(end-1,2:end-1,2:end-1); utb5 = Ut(2:end-1,end-1,2:end-1); utb6 = Ut(2:end-1,2:end-1,end-1); 
%        UtB_1 = [utb1(:); utb2(:); utb3(:); utb4(:); utb5(:); utb6(:)];                     
%        Bmax = max((h^2/(1i*a))*UtB_1./UB_1);
%        clear UtB_1 UB_1 Ut ub1 ub2 ub3 ub4 ub5 ub6 utb1 utb2 utb3 utb4 utb5 utb6;
%     elseif(BC==3)
%        u1 = U(1,:,:);   u2 = U(:,1,:);   u3 = U(:,:,1);
%        u4 = U(end,:,:); u5 = U(:,end,:); u6 = U(:,:,end);
%        UB = [u1(:); u2(:); u3(:); u4(:); u5(:); u6(:)];           
%        v1 = V(1,:,:);   v2 = V(:,1,:);   v3 = V(:,:,1);
%        v4 = V(end,:,:); v5 = V(:,end,:); v6 = V(:,:,end);
%        VB = [v1(:); v2(:); v3(:); v4(:); v5(:); v6(:)];  
%        Bmax = max((h^2/a)*(s*UB.*conj(UB) - VB));
%        clear VB UB u1 u2 u3 u4 u5 u6 v1 v2 v3 v4 v5 v6;
%    end    
%    %Now compute full k bound:
%    kfull = (h^2/a)*sqrt(8)/max(abs(Bmax),max(abs(Lmaxes)));       
%    disp(['kmax (linear): ',num2str(klin)]);
%    disp(['kmax   (full): ',num2str(kfull)]);
%    if(kfull<klin)
%       k2 = 0.91*kfull;
%    else
%       k2 = 0.91*klin; 
%    end 
%    disp(['k2      (used): ',num2str(k2)]);
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%PLOT%%%SOLUTION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(plot_vorticity==1 || track_ring==1)                 
     %Compute Vorticity:       
     [w_x,w_y,w_z,W] = rmc_vorticity3D(U,1/(h*2),vorticity_eps);
end

if(plot_3D_vol==1)
    if(plot_modulus2==1)
        if(smooth_3d==0)
           plot_cube_mod2.cdata = U.*conj(U); %Compute density
        else
           plot_cube_mod2.cdata = smooth3(U.*conj(U));
        end
        if(chunk_count==1 && high_qual==0), plot_cube_mod2.texture = '2D'; end;
        plot_cube_mod2 = vol3d(plot_cube_mod2);        
        %time_txt   = ['Time:      ',num2str((n*k),'%.2f')];
        %set(plot_label,'String',time_txt);
        set(fig_cube_mod2, 'Name',['3D NLSE  t = ',num2str((n*k),'%.2f')]);        
        if(save_images==1)            
            set(fig_cube_mod2,'InvertHardcopy','off')
            set(fig_cube_mod2, 'PaperPositionMode', 'auto'); 
            print('-djpeg','-r100','-opengl',[fn,'_n',num2str(n), '_MOD2_3DVOL_t=' strrep(num2str(n*k),'.','')],['-f',num2str(fig_cube_mod2)]);   
        end
        if(save_movies==1 || save_movies==3)
            gifname = [fn,'_MOD2_3DVOL.gif'];
            I = getframe(fig_cube_mod2);
            I = frame2im(I);
            [XX, map] = rgb2ind(I, 128);
            imwrite(XX, map, gifname, 'GIF', 'WriteMode', 'append', 'DelayTime', 0);
        end
        if(save_movies==2 || save_movies==3)
            mov_cube_mod2 = addframe(mov_cube_mod2,getframe(fig_cube_mod2));  
        end
        if(chunk_count==numframes)
            plot_cube_mod2.texture = '3D';     
            plot_cube_mod2 = vol3d(plot_cube_mod2);
        end
        
    end%plot mod2        
    if(plot_vorticity==1)         
       if(smooth_3d==0)
          plot_cube_vort.cdata = W;
       else
          plot_cube_vort.cdata = smooth3(W); 
       end       
       if(chunk_count==1 && high_qual==0), plot_cube_vort.texture = '2D'; end;  
       plot_cube_vort = vol3d(plot_cube_vort); %Refresh cube       
       set(fig_cube_vort, 'Name',['3D NLSE Vorticity t = ',num2str((n*k),'%.2f')]);
       if(save_movies==1 || save_movies==3)
            gifname = [fn,'_VORT_3DVOL.gif'];
            I = getframe(fig_cube_vort);
            I = frame2im(I);
            [XX, map] = rgb2ind(I, 128);
            imwrite(XX, map, gifname, 'GIF', 'WriteMode', 'append', 'DelayTime', 0);
       end
       if(save_movies==2 || save_movies==3)
            mov_cube_vort = addframe(mov_cube_vort,getframe(fig_cube_vort));  
       end   
       if(save_images==1)
            set(fig_cube_vort,'InvertHardcopy','off')
            set(fig_cube_vort, 'PaperPositionMode', 'auto');       
            fignum = ['-f',num2str(fig_cube_vort)];
            print('-djpeg','-r100',[fn,'_n',num2str(n), '_T', strrep(num2str(n*k),'.',''), '_VORT_3DVOL'],fignum);  
       end
       if(chunk_count==numframes)
            plot_cube_vort.texture = '3D';     
            plot_cube_vort = vol3d(plot_cube_vort);
       end     
    end%vort  
end%plot vol3D   

if(plot_2D_cuts==1)      
   if(plot_modulus2==1)
       set(plot_cuts_mod2_XY,'Cdata',squeeze(U(:,:,floor(zres/2)).*conj(U(:,:,floor(zres/2)))));
      % time_txt   = ['Time:      ',num2str((n*k),'%.2f')];
      % set(plot_label,'String',time_txt);
       set(plot_cuts_mod2_XZ,'Cdata',squeeze(U(floor(yres/2),:,:).*conj(U(floor(yres/2),:,:))));
       set(plot_cuts_mod2_YZ,'Cdata',squeeze(U(:,floor(xres/2),:).*conj(U(:,floor(xres/2),:))));
       set(fig_cuts_mod2, 'Name',['3D NLSE Mod2 t = ',num2str((n*k),'%.2f')]);
       if(save_movies==1 || save_movies==3)    
           gifname = [fn,'_MOD2_2DCUTS.gif'];
           I = getframe(fig_cuts_mod2);
           I = frame2im(I);
           [XX, map] = rgb2ind(I, 128);
           imwrite(XX, map, gifname, 'GIF', 'WriteMode', 'append', 'DelayTime', 0);
       end
       if(save_movies==2 || save_movies==3)
            mov_cuts_mod2 = addframe(mov_cuts_mod2,getframe(fig_cuts_mod2));         
       end
       if(save_images==1)
            set(fig_cuts_mod2,'InvertHardcopy','off')
            set(fig_cuts_mod2, 'PaperPositionMode', 'auto'); 
            fignum = ['-f',num2str(fig_cuts_mod2)];
            print('-depsc',strrep([fn,'_n',num2str(n),'_T', strrep(num2str(n*k),'.',''), '_MOD2_2DCUTS'],'.',''),fignum);  
       end
   end%plot_mod2
   
   if(plot_phase==1)
       set(plot_cuts_phase_XY,'Cdata',squeeze(angle(U(:,:,floor(zres/2)))));
       set(plot_cuts_phase_XZ,'Cdata',squeeze(angle(U(floor(yres/2),:,:))));
       set(plot_cuts_phase_YZ,'Cdata',squeeze(angle(U(:,floor(xres/2),:))));
       set(fig_cuts_phase, 'Name',['3D NLSE Phase t = ',num2str((n*k),'%.2f')]);       
       if(save_movies==1 || save_movies==3)
           gifname = [fn,'_PHASE_2DCUTS.gif'];
           I = getframe(fig_cuts_phase);
           I = frame2im(I);
           [XX, map] = rgb2ind(I, 128);
           imwrite(XX, map, gifname, 'GIF', 'WriteMode', 'append', 'DelayTime', 0);
       end
       if(save_movies==2 || save_movies==3)
           mov_cuts_phase = addframe(mov_cuts_phase,getframe(fig_cuts_phase)); 
       end       
       if(save_images==1)
            set(fig_cuts_phase,'InvertHardcopy','off')
            set(fig_cuts_phase, 'PaperPositionMode', 'auto'); 
            fignum = ['-f',num2str(fig_cuts_phase)];
            print('-depsc',strrep([fn,'_n',num2str(n),'_T', strrep(num2str(n*k),'.',''), '_PHASE_2DCUTS'],'.',''),fignum);  
       end
   end   
   
   if(plot_vorticity==1)
        set(plot_cuts_vort_XY,'Cdata',squeeze(W(:,:,floor(zres/2))));
        set(plot_cuts_vort_XZ,'Cdata',squeeze(W(floor(yres/2),:,:)));
        set(plot_cuts_vort_YZ,'Cdata',squeeze(W(:,floor(xres/2),:)));
        set(fig_cuts_vort, 'Name',['3D NLSE Vorticity t = ',num2str((n*k),'%.2f')]); 
        if(save_movies==1 || save_movies==3)
           gifname = [fn,'_VORT_2DCUTS.gif'];
           I = getframe(fig_cuts_vort);
           I = frame2im(I);
           [XX, map] = rgb2ind(I, 128);
           imwrite(XX, map, gifname, 'GIF', 'WriteMode', 'append', 'DelayTime', 0);
        end
        if(save_movies==2 || save_movies==3)
           mov_cuts_vort = addframe(mov_cuts_vort,getframe(fig_cuts_vort));        
        end     
        if(save_images==1)
            set(fig_cuts_vort,'InvertHardcopy','off')
            set(fig_cuts_vort, 'PaperPositionMode', 'auto'); 
            fignum = ['-f',num2str(fig_cuts_vort)];
            print('-depsc',strrep([fn,'_n',num2str(n),'_T', strrep(num2str(n*k),'.',''),'_VORT_2DCUTS'],'.',''),fignum);  
        end
   end
end%plot2Dcuts

%Refresh plots if no movie saved (movie auto-refreshes)
if(save_movies==0 && (plot_2D_cuts==1 || plot_3D_vol==1))
    drawnow;
end
       
%Display single point for error comparason/convergance tests: 
if(disp_pt_data==1)
    disp(['3D NLSE  t = ',num2str((n*k),'%.2f'),'U(5,5,5) = ',num2str(U(5,5,5))]);
end

%%%Analysis Tools%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
calcn           = calcn+1;
plottime(calcn) = n*k;   

%Calculate mass:
if(calc_mass==1)
   mass(calcn) = sum(U(:).*conj(U(:))).*h^2;     
   delta_mass = (abs(mass(1)-mass(calcn))/mass(1))*100;
end

%Record error for linear test (IC=1)
if(IC==1)
    Ureal = exp(-(X.^2 + Y.^2 + Z.^2)/(2*a)).*exp(-1i*3*k*n);
    errorvecr(calcn) = sqrt(sum((real(Ureal(:)) - real(U(:))).^2)/length(U(:)));
    errorveci(calcn) = sqrt(sum((imag(Ureal(:)) - imag(U(:))).^2)/length(U(:)));
    errorvecm(calcn) = sqrt(sum((abs(Ureal(:)).^2  -  abs(U(:)).^2).^2)/length(U(:)));
end

%Track vortex ring's z-position
if(track_ring==1)      
    
    WZ   = squeeze(W(:,floor(xres/2),:));
    Yvec = squeeze(Y(:,floor(xres/2),1));
    Zvec = squeeze(Z(1,floor(xres/2),:));
    Y2D  = squeeze(Y(:,floor(xres/2),:));
    Z2D  = squeeze(Z(:,floor(xres/2),:));
    
    sizeu = size(WZ);           
    scan_radius = 2*sqrt(abs(a/OM));
    sr_i        = floor(scan_radius/h); 
    
    for vi=1:num_vort    
       if(calcn==1)   
           yv1(vi,calcn) = vortrings(4,vi) - vortrings(13,vi);
           zv1(vi,calcn) = vortrings(5,vi);
           yv2(vi,calcn) = vortrings(4,vi) + vortrings(13,vi);
           zv2(vi,calcn) = vortrings(5,vi);
                      
           [tmp,yiv1(vi,calcn)] = min(abs(Yvec-yv1(vi,calcn)));
           [tmp,ziv1(vi,calcn)] = min(abs(Zvec-zv1(vi,calcn)));
           [tmp,yiv2(vi,calcn)] = min(abs(Yvec-yv2(vi,calcn)));
           [tmp,ziv2(vi,calcn)] = min(abs(Zvec-zv2(vi,calcn)));        
           
           yv1(vi,calcn) = Yvec(yiv1(vi,calcn));
           yv2(vi,calcn) = Yvec(yiv2(vi,calcn));
           zv1(vi,calcn) = Zvec(ziv1(vi,calcn));
           zv2(vi,calcn) = Zvec(ziv2(vi,calcn));    
       else
       
       yiv1min = yiv1(vi,calcn-1)-sr_i;
       yiv2min = yiv2(vi,calcn-1)-sr_i;
       ziv1min = ziv1(vi,calcn-1)-sr_i;
       ziv2min = ziv2(vi,calcn-1)-sr_i;     
       
       yiv1max = yiv1(vi,calcn-1)+sr_i;
       yiv2max = yiv2(vi,calcn-1)+sr_i;
       ziv1max = ziv1(vi,calcn-1)+sr_i;
       ziv2max = ziv2(vi,calcn-1)+sr_i;
       
       %Check if simulation has hit boundary:           
       if( ((yiv1min < 1        || yiv2min < 1        ) || (ziv1min < 1        || ziv2min < 1       )) ||...
           ((yiv1max > sizeu(1) || yiv2max > sizeu(1) ) || (ziv1max > sizeu(2) || ziv2max > sizeu(2))) )
           disp('A vortex ring is too close to boundary to track, ending tracking.');        
           yiv1(vi,calcn) = yiv1(vi,calcn-1);
           yiv2(vi,calcn) = yiv2(vi,calcn-1);
           ziv1(vi,calcn) = ziv1(vi,calcn-1);
           ziv2(vi,calcn) = ziv2(vi,calcn-1);
       else                      
          [yiv1(vi,calcn), ziv1(vi,calcn)] = find(WZ(yiv1min:yiv1max,   ...
                                                  ziv1min:ziv1max) == ...
                                       max(max(WZ(yiv1min:yiv1max,   ...
                                                  ziv1min:ziv1max))));                                 
          [yiv2(vi,calcn), ziv2(vi,calcn)] = find(WZ(yiv2min:yiv2max,   ...
                                                  ziv2min:ziv2max) == ...
                                       max(max(WZ(yiv2min:yiv2max,   ...
                                                  ziv2min:ziv2max))));        
          yiv1(vi,calcn) = yiv1min + yiv1(vi,calcn)-1;
          yiv2(vi,calcn) = yiv2min + yiv2(vi,calcn)-1;
          ziv1(vi,calcn) = ziv1min + ziv1(vi,calcn)-1;
          ziv2(vi,calcn) = ziv2min + ziv2(vi,calcn)-1; 
       end %no error
       
       yv1(vi,calcn)  = Yvec(yiv1(vi,calcn));
       yv2(vi,calcn)  = Yvec(yiv2(vi,calcn));
       zv1(vi,calcn)  = Zvec(ziv1(vi,calcn));
       zv2(vi,calcn)  = Zvec(ziv2(vi,calcn));  
       end %calcn=1 
       
       %Now compute new search grid for center of mass calculations:
       yiv1min = yiv1(vi,calcn)-sr_i;
       yiv2min = yiv2(vi,calcn)-sr_i;
       ziv1min = ziv1(vi,calcn)-sr_i;
       ziv2min = ziv2(vi,calcn)-sr_i;     
       
       yiv1max = yiv1(vi,calcn)+sr_i;
       yiv2max = yiv2(vi,calcn)+sr_i;
       ziv1max = ziv1(vi,calcn)+sr_i;
       ziv2max = ziv2(vi,calcn)+sr_i;
       if( ((yiv1min < 1        || yiv2min < 1        ) || (ziv1min < 1        || ziv2min < 1       )) ||...
           ((yiv1max > sizeu(1) || yiv2max > sizeu(1) ) || (ziv1max > sizeu(2) || ziv2max > sizeu(2))) )
           disp('A vortex ring is too close to boundary to track, ending tracking.');  
           yiv1(vi,calcn) = yiv1(vi,calcn-1);
           yiv2(vi,calcn) = yiv2(vi,calcn-1);
           ziv1(vi,calcn) = ziv1(vi,calcn-1);
           ziv2(vi,calcn) = ziv2(vi,calcn-1);             
           yv1(vi,calcn)  = Yvec(yiv1(vi,calcn));
           yv2(vi,calcn)  = Yvec(yiv2(vi,calcn));
           zv1(vi,calcn)  = Zvec(ziv1(vi,calcn));
           zv2(vi,calcn)  = Zvec(ziv2(vi,calcn));             
           yiv1min = yiv1(vi,calcn)-sr_i;
           yiv2min = yiv2(vi,calcn)-sr_i;
           ziv1min = ziv1(vi,calcn)-sr_i;
           ziv2min = ziv2(vi,calcn)-sr_i;            
           yiv1max = yiv1(vi,calcn)+sr_i;
           yiv2max = yiv2(vi,calcn)+sr_i;
           ziv1max = ziv1(vi,calcn)+sr_i;
           ziv2max = ziv2(vi,calcn)+sr_i;
       end %error      
       
       Mv1 = sum(sum(WZ(yiv1min:yiv1max,ziv1min:ziv1max))*h^2);
       Mv2 = sum(sum(WZ(yiv2min:yiv2max,ziv2min:ziv2max))*h^2);
       
       v_trackY1(vi,calcn)  = h^2*sum(sum(Y2D(yiv1min:yiv1max,ziv1min:ziv1max).* ...
                                           WZ(yiv1min:yiv1max,ziv1min:ziv1max)))/Mv1;       
       v_trackZ1(vi,calcn)  = h^2*sum(sum(Z2D(yiv1min:yiv1max,ziv1min:ziv1max).* ...
                                           WZ(yiv1min:yiv1max,ziv1min:ziv1max)))/Mv1;  
       v_trackY2(vi,calcn)  = h^2*sum(sum(Y2D(yiv2min:yiv2max,ziv2min:ziv2max).* ...
                                           WZ(yiv2min:yiv2max,ziv2min:ziv2max)))/Mv2;       
       v_trackZ2(vi,calcn)  = h^2*sum(sum(Z2D(yiv2min:yiv2max,ziv2min:ziv2max).* ...
                                           WZ(yiv2min:yiv2max,ziv2min:ziv2max)))/Mv2; 
      
    end
    
    %Co-planer VR merge tests:
    if((num_vort==2 && length(qv)>1) && coplane==1)
          if(mergehappened==0)
          if(v_trackY1(1,calcn)==v_trackY1(2,calcn) || v_trackY1(1,calcn)==v_trackY2(2,calcn))
              mergetime(qi) = n*k;
              mergez(qi) = v_trackZ1(1,calcn);
              mergehappened = 1;
          end
          if(v_trackY2(1,calcn)==v_trackY2(2,calcn) || v_trackY2(1,calcn)==v_trackY1(2,calcn))
              mergetime(qi) = n*k;
              mergez(qi) = v_trackZ2(1,calcn);
              mergehappened = 1;
          end
          end
    end
end %Trackvortrings

if(mergehappened==1)
    break;
end

%Update waitbar:
if(show_waitbar==1)
    time4frame      = toc(tchunk);
    totsecremaining = (numframes-chunk_count)*time4frame;
    minremaining    = floor(totsecremaining/60);
    secremaining    = floor(60*(totsecremaining/60 - minremaining));
    set(fig_waitbar, 'Name',['Simulation ',num2str((n/steps)*100),'% Complete']);
    waitbar(n/steps,fig_waitbar,['Estimated time to completion ',num2str(minremaining,'%03d'),...
    ':',num2str(secremaining,'%02d')]);  
end

end %chunk-count
%Simulation is now over!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Simulation is now over!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Simulation is now over!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Simulation is now over!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Simulation is now over!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Clear non-mex RK4 matrices if used:
if(method>10)
   clear Uk_tmp;
   clear k_tot;
   clear functions;
end  

%How long did this take?
totaltime(run_i) = toc(ttotal);
totalcomp(run_i) = tcomp;

if(runs>1)
    disp(['Run ',num2str(run_i),':   ',num2str(totaltime(run_i)),'  ',...
                      num2str(totalcomp(run_i))]);
end

%Close waitbar figure and movie file:
if(show_waitbar==1)
    close(fig_waitbar);
end
if(save_movies==2 || save_movies==3)
    if(plot_3D_vol==1)
       if(plot_modulus2==1),  mov_cube_mod2 = close(mov_cube_mod2);end;
       if(plot_vorticity==1), mov_cube_vort = close(mov_cube_vort);end;
    end
    if(plot_2D_cuts==1)
       if(plot_modulus2==1),  mov_cuts_mod2  = close(mov_cuts_mod2); end;
       if(plot_phase==1),     mov_cuts_phase = close(mov_cuts_phase);end
       if(plot_vorticity==1), mov_cuts_vort  = close(mov_cuts_vort); end;       
    end  
end%save_mov

end %Repeated runs complete %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Compute averaged times and output result:
tottime   = mean(totaltime);
tcomp     = mean(totalcomp);

if(runs>1)
    ttstd    = std(totaltime) ;
    tcstd    = std(tcomp);    
    disp(['Simulation   Total Time: ',num2str(tottime),...
           ' (std:',num2str(ttstd),') seconds.']);
    disp(['Simulation Compute Time: ',num2str(tcomp),...
        ' (std:',num2str(tcstd),') seconds.']);    
else
    disp(['Simulation   Total Time: ',num2str(tottime),' seconds.']);
    disp(['Simulation Compute Time: ',num2str(tcomp),' seconds.']);   
end

%Plot results and output:
if(calc_mass == 1)
   disp('Now plotting mass....');
   disp(['Total Mass:',num2str(mass(end))]);
   figure(fig_count);  fig_count = fig_count+1;
   plot(plottime,(mass - mass(1))./mass(1),'-');
%   title_txt = ['\DeltaMass = ',num2str(delta_mass,'%.4f'),'%','\newline', ...
%              title_txt_params];
%   title(title_txt);   
   grid on;
   xlabel('Time');   ylabel('Mass');
end

if(track_ring==1)
   %Setup velocity vectors:
   VY1 = zeros(size(v_trackY1));
   VZ1 = VY1;   VY2 = VY1;   VZ2 = VY1; VZC = VY1;
   
   %Get time step of plots:
   plot_dt = plottime(2)-plottime(1);
    
   %Calculate velocities using second-order differencing
   for vi=1:num_vort
       VY1(vi,2:end-1) = (v_trackY1(vi,3:end) - v_trackY1(vi,1:end-2))./(2*plot_dt);
       VY1(vi,1)       = (v_trackY1(vi,2)     - v_trackY1(vi,1))./plot_dt;
       VY1(vi,end)     = (v_trackY1(vi,end)   - v_trackY1(vi,end-1))./plot_dt;       
       VY2(vi,2:end-1) = (v_trackY2(vi,3:end) - v_trackY2(vi,1:end-2))./(2*plot_dt);
       VY2(vi,1)       = (v_trackY2(vi,2)     - v_trackY2(vi,1))./plot_dt;
       VY2(vi,end)     = (v_trackY2(vi,end)   - v_trackY2(vi,end-1))./plot_dt;       
       VZ1(vi,2:end-1) = (v_trackZ1(vi,3:end) - v_trackZ1(vi,1:end-2))./(2*plot_dt);
       VZ1(vi,1)       = (v_trackZ1(vi,2)     - v_trackZ1(vi,1))./plot_dt;
       VZ1(vi,end)     = (v_trackZ1(vi,end)   - v_trackZ1(vi,end-1))./plot_dt;       
       VZ2(vi,2:end-1) = (v_trackZ2(vi,3:end) - v_trackZ2(vi,1:end-2))./(2*plot_dt);
       VZ2(vi,1)       = (v_trackZ2(vi,2)     - v_trackZ2(vi,1))./plot_dt;
       VZ2(vi,end)     = (v_trackZ2(vi,end)   - v_trackZ2(vi,end-1))./plot_dt;
       
       v_trackZC(vi,:) = (v_trackZ1(vi,:)+v_trackZ2(vi,:))/2;
       VZC(vi,2:end-1) = (v_trackZC(vi,3:end) - v_trackZC(vi,1:end-2))./(2*plot_dt);
       VZC(vi,1)       = (v_trackZC(vi,2)     - v_trackZC(vi,1))./plot_dt;
       VZC(vi,end)     = (v_trackZC(vi,end)   - v_trackZC(vi,end-1))./plot_dt;       
   end     
   
   if(length(rmaxv)==1)
 
     fnbase = ['_q',num2str(q),'_A',num2str(ea),'_d',num2str(d),'_a',num2str(a),'_s',num2str(s),'_OM',num2str(OM),...
             '_T',num2str(endt),'_M',num2str(method),'_P',num2str(precision),'_h',num2str(h),'_k',num2str(k)];
     fnbase = strrep(fnbase,'.','o');  
       
       
       
%Plot ring center-position:
   figcenters = figure(fig_count);  fig_count = fig_count+1;  
   hold on;
   for vi=1:num_vort
       plot((v_trackY1(vi,:)+v_trackY2(vi,:))/2,(v_trackZ1(vi,:)+v_trackZ2(vi,:))/2,'ok','LineWidth',2)         
   end
   hold off;
   title('Ring center position');
   axis equal;
   xlabel('Y')
   ylabel('Z')
   axis([ymin ymax zmin zmax]);   
  
   
   
   %Plot ring center z versus time:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   figzct = figure(fig_count);  fig_count = fig_count+1;  
   for vi=1:num_vort
       plot(plottime, v_trackZC(vi,:),'-k','LineWidth',5)      
       if(vi==1); hold on; end;
   end
   hold off;
   set(gcf,'Name','Vortex ring z-center versus time');
   set(gca,'Fontsize',fsize);
   maxaxisy = max(v_trackZC(:)) + 0.1*abs(max(v_trackZC(:))-min(v_trackZC(:)));
   minaxisy = min(v_trackZC(:)) - 0.1*abs(max(v_trackZC(:))-min(v_trackZC(:)));
   axis([0 max(plottime) minaxisy maxaxisy]);
   xlabel('Time','Fontsize',fsize);
   ylabel('z_c(t)','Fontsize',fsize);
   axis([0 max(plottime) zmin zmax]);   
   
  
       
   %Plot ring radius vs time:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   figrad = figure(fig_count);  fig_count = fig_count+1;
   for vi=1:num_vort
      v_trackd(vi,:) =  abs(v_trackY1(vi,:) - v_trackY2(vi,:))./2;
      plot(plottime,v_trackd(vi,:),'-b','LineWidth',5);
      if(vi==1); hold on; end;
      %plot(plottime,abs(yv1(vi,:) - yv2(vi,:))./2,'sr','LineWidth',2);
   end
   hold off;
   maxaxisy = max(v_trackd(:)) + 0.1*abs(max(v_trackd(:))-min(v_trackd(:)));
   minaxisy = min(v_trackd(:)) - 0.1*abs(max(v_trackd(:))-min(v_trackd(:)));
   axis([0 max(plottime) minaxisy maxaxisy]);
   set(gcf,'Name','Vortex ring radius versus time');
   set(gca,'Fontsize',fsize);
   xlabel('Time','Fontsize',fsize);
   ylabel('d','Fontsize',fsize);
   
   if(save_plots==1) 
      figure(figrad)
      fndvt = ['DvsT',fnbase];
      set(gcf, 'PaperPositionMode', 'auto');       
      print('-djpeg','-r100',fndvt,['-f',num2str(figrad)]); 
      print('-depsc','-r100',fndvt,['-f',num2str(figrad)]); 
      saveas(figrad,[fndvt,'.fig']);   
   end
   
   %Plot positional graph Y vs Z:   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   figpos = figure(fig_count);  fig_count = fig_count+1;
   symbol_list = ['o','s'];
   qs = 1; %skip distance for better plot
   for vi=1:num_vort
       if(vortrings(vi,1)>0)
           marker1 = [symbol_list(vi) 'b'];           marker2 = [symbol_list(vi) 'k'];
       else
           marker2 = [symbol_list(vi) 'b'];           marker1 = [symbol_list(vi) 'k']; 
       end       
       plot(v_trackY1(vi,1:qs:end),v_trackZ1(vi,1:qs:end),marker1,'LineWidth',2);
        if(vi==1); hold on; end
       plot(v_trackY2(vi,1:qs:end),v_trackZ2(vi,1:qs:end),marker2,'LineWidth',2);                
 end   
%    if(num_vort==2)
%       plot((v_trackY1(1,:)+v_trackY2(1,:))/2,(v_trackZ1(1,:)+v_trackZ2(1,:))/2,'-r','LineWidth',2)
%       plot((v_trackY1(1,:)+v_trackY1(2,:))/2,(v_trackZ1(1,:)+v_trackZ1(2,:))/2,':r','LineWidth',2)
%       plot((v_trackY1(2,:)+v_trackY2(2,:))/2,(v_trackZ1(2,:)+v_trackZ2(2,:))/2,'-g','LineWidth',2)
%       plot((v_trackY2(2,:)+v_trackY2(1,:))/2,(v_trackZ2(2,:)+v_trackZ2(1,:))/2,':g','LineWidth',2)
%    end 
   hold off;
   set(gcf,'Name','Vortex ring position trace');   
   maxaxisz = max([v_trackZ1(:);v_trackZ2(:)]) + 0.3*abs(max([v_trackZ1(:);v_trackZ2(:)])-min([v_trackZ1(:);v_trackZ2(:)]));
   minaxisz = min([v_trackZ1(:);v_trackZ2(:)]) - 0.3*abs(max([v_trackZ1(:);v_trackZ2(:)])-min([v_trackZ1(:);v_trackZ2(:)]));
   maxaxisy = max([v_trackY1(:);v_trackY2(:)]) + 0.3*abs(max([v_trackY1(:);v_trackY2(:)])-min([v_trackY1(:);v_trackY2(:)]));
   minaxisy = min([v_trackY1(:);v_trackY2(:)]) - 0.3*abs(max([v_trackY1(:);v_trackY2(:)])-min([v_trackY1(:);v_trackY2(:)]));
   axis('equal');
   axis([ymin ymax zmin zmax]);   
   set(gca,'Fontsize',fsize);
   xlabel('y(t)','Fontsize',fsize);
   ylabel('z(t)','Fontsize',fsize);
   
   if(save_plots==1) 
      figure(figpos)
      fnpos = ['ZvsY',fnbase];
      set(gcf, 'PaperPositionMode', 'auto');       
      print('-djpeg','-r100',fnpos,['-f',num2str(figpos)]); 
      print('-depsc','-r100',fnpos,['-f',num2str(figpos)]); 
      saveas(figpos,[fnpos,'.fig']);   
   end
   
   %Plot z-position vs time%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   figzct = figure(fig_count);  fig_count = fig_count+1;
   qs = 1; %skip distance for better plot
   for vi=1:num_vort
       if(vi==1); cl='-b'; end;
       if(vi==2); cl='-r'; end;    
       plot(plottime,v_trackZC(vi,:),'-b','LineWidth',4);
        if(vi==1); hold on; end
      % plot(plottime,v_trackZ2(vi,:),'-r','LineWidth',2);   
      % plot(plottime,v_trackZC(vi,:),'-k','LineWidth',5);
       %plot(plottime,zv1(vi,:),marker1,'LineWidth',2);
       %plot(plottime,zv2(vi,:),marker2,'LineWidth',2);
   end   
   if(num_vort==2)
      plot(plottime,(v_trackZC(1,:)+v_trackZC(2,:))./2,'-k','LineWidth',4);
   end
   hold off;
   set(gcf,'Name','Vortex Ring z-positions vs time');
   maxaxis = max([v_trackZ1(:);v_trackZ2(:)]) + 0.1*abs(max([v_trackZ1(:);v_trackZ2(:)])-min([v_trackZ1(:);v_trackZ2(:)]));
   minaxis = min([v_trackZ1(:);v_trackZ2(:)]) - 0.1*abs(max([v_trackZ1(:);v_trackZ2(:)])-min([v_trackZ1(:);v_trackZ2(:)]));
   axis([0 max(plottime) minaxis maxaxis]);  
   set(gca,'Fontsize',fsize);
   xlabel('Time','Fontsize',fsize);
   ylabel('z(t)','Fontsize',fsize);
   
   if(save_plots==1) 
      figure(figzct)
      fnzct = ['ZvsT',fnbase];
      set(gcf, 'PaperPositionMode', 'auto');       
      print('-djpeg','-r100',fnzct,['-f',num2str(figzct)]); 
      print('-depsc','-r100',fnzct,['-f',num2str(figzct)]); 
      saveas(figzct,[fnzct,'.fig']);   
   end
      
   %Plot y-position vs time %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   figyvt = figure(fig_count);  fig_count = fig_count+1;
   symbol_list = ['o','s'];
   qs = 1; %skip distance for better plot
   for vi=1:num_vort
       plot(plottime,v_trackY1(vi,:),'-','LineWidth',5);
        if(vi==1); hold on; end
       plot(plottime,v_trackY2(vi,:),'-','LineWidth',5);   
       %plot(plottime,yv1(vi,:),marker1,'LineWidth',2);
       %plot(plottime,yv2(vi,:),marker2,'LineWidth',2);
   end
   hold off;
   set(gcf,'Name','Vortex Ring y-positions vs time');
   maxaxis = max([v_trackY1(:);v_trackY2(:)]) + 0.1*abs(max([v_trackY1(:);v_trackY2(:)])-min([v_trackY1(:);v_trackY2(:)]));
   minaxis = min([v_trackY1(:);v_trackY2(:)]) - 0.1*abs(max([v_trackY1(:);v_trackY2(:)])-min([v_trackY1(:);v_trackY2(:)]));
   axis([0 max(plottime) minaxis maxaxis]);  
   set(gca,'Fontsize',fsize);
   xlabel('Time','Fontsize',fsize);
   ylabel('y(t)','Fontsize',fsize)
    
   if(save_plots==1) 
      figure(figyvt)
      fnzct = ['YvsT',fnbase];
      set(gcf, 'PaperPositionMode', 'auto');       
      print('-djpeg','-r100',fnzct,['-f',num2str(figyvt)]); 
      print('-depsc','-r100',fnzct,['-f',num2str(figyvt)]); 
      saveas(figyvt,[fnzct,'.fig']);   
   end
   
   %Display merge time and position for co-plane merge tests:
   if(coplane==1)
      mergetime'
      mergez'
   end   
   
   %Plot z-velocity%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   figure(fig_count);  fig_count = fig_count+1;
   halftimei = floor(length(plottime)/2);  

   plot(plottime,VZC(1,:),'ob','LineWidth',3); 
   hold on;
   plot(plottime,c*ones(size(plottime)),'-r','LineWidth',5);
   hold off;
   set(gcf,'Name','Vortex ring centered z-velocity');
   maxaxis = max([VZC(1,:) c]) + 0.1*abs(max([VZC(1,:) c])-min([VZC(1,:) c]));
   minaxis = min([VZC(1,:) c]) - 0.1*abs(max([VZC(1,:) c])-min([VZC(1,:) c]));
   axis([0 max(plottime) minaxis maxaxis]);
   set(gca,'Fontsize',fsize);
   xlabel('Time','Fontsize',fsize);
   ylabel('c = dz/dt','Fontsize',fsize);
%    
   end %length(rmax)==1
   
   %Compute z-velocity for leapfrog test:
   if(num_vort==2)
      v_trackZCC(1,:) = (v_trackZC(1,:)+v_trackZC(2,:))./2;     
   end
   
   %Compute z-velocity through linear fit:
   if(num_vort==2)
      zvelCoMtmp         = ezfit(plottime,v_trackZCC(1,:),'poly1');
   else
      zvelCoMtmp         = ezfit(plottime,v_trackZC(1,:),'poly1');
   end
   
   zvelCoM(rmaxi,bi)  = zvelCoMtmp.m(2);   
   zvelgridtmp        = ezfit(plottime,zv1(1,:),'poly1');
   zvelgrid(rmaxi,bi) = zvelgridtmp.m(2);
   
   zvelCoMerr(rmaxi,bi)  =  100*(zvelCoM(rmaxi,bi)-c)/c;
   zvelgriderr(rmaxi,bi) =  100*(zvelgrid(rmaxi,bi)-c)/c;   
   
   %Record velocity for quadrupole tests:
   zvelAv(ai,di) = zvelCoM(rmaxi,bi);
   
   if(length(Av)>1)
   figure(figrad)
   title(['c: ',num2str(zvelAv(ai,di))]);
   end
   
   disp('VR Velocity');
   disp(['Pred: ',num2str(c)]);
   disp(['CofM: ',num2str(zvelCoM(rmaxi,bi)), '  Percent Error: ',num2str(zvelCoMerr(rmaxi,bi)),'%']);
   disp(['Grid: ',num2str(zvelgrid(rmaxi,bi)),'  Percent Error: ',num2str(zvelgriderr(rmaxi,bi)),'%']);  
   
end

%Linear example errors:
if(IC==1)
    fig_err = figure(fig_count);
    fig_count = fig_count +1;
    set(fig_err, 'Name','Error of Run','NumberTitle','off');
    plot(plottime,errorveci,'r'); hold on;
    plot(plottime,errorvecr,'b');
    plot(plottime,errorvecm,'k'); hold off;
    errorsr(hi) = mean(errorvecr);
    errorsi(hi) = mean(errorveci);
    errorsm(hi) = mean(errorvecm);
    if(hi>1)
        mr(hi-1) = abs(log(errorsr(hi-1))-log(errorsr(hi)))./ ...
                   abs(log(hv(hi-1)) - log(h));
        mi(hi-1) = abs(log(errorsi(hi-1))-log(errorsi(hi)))./ ...
                   abs(log(hv(hi-1)) - log(h)); 
        mm(hi-1) = abs(log(errorsm(hi-1))-log(errorsm(hi)))./ ...
                   abs(log(hv(hi-1)) - log(h)); 
        disp(['Step ',num2str(hi),' of ',num2str(length(hv)),' done. h:',num2str(h),...
        ' Error_ave:',num2str((errorsr(hi)+errorsi(hi))/2),...
        ' mave:',num2str( (mr(hi-1)+mi(hi-1))/2  )]);
    else
        if(length(hv)>1)
        disp(['Step ',num2str(hi),' of ',num2str(length(hv)),' done. h:',num2str(h),...
        ' Error_ave:',num2str((errorsr(hi)+errorsi(hi))/2)]);
        end
    end         
end

end %hv stepsizes loop for order tests done %%%%%%%%%%%%%%%%%%%%%%%%%

%Output results in latex table format:
if(IC==1)
    h_str       = num2str(hv(1));
    errorsr_str = num2str(errorsr(1));    errorsi_str = num2str(errorsi(1));
    errorsm_str = num2str(errorsm(1));
    mr_str = '--';    mi_str = '--';    mm_str = '--';
    for st = 2:length(hv)
        h_str = [h_str, ' & ', num2str(hv(st))];
        errorsr_str = [errorsr_str, ' & ', num2str(errorsr(st))];
        errorsi_str = [errorsi_str, ' & ', num2str(errorsi(st))];
        errorsm_str = [errorsm_str, ' & ', num2str(errorsm(st))];   
        mr_str = [mr_str, ' & ', num2str(mr(st-1))];
        mi_str = [mi_str, ' & ', num2str(mi(st-1))];
        mm_str = [mm_str, ' & ', num2str(mm(st-1))];
    end
    disp('-----------------------------------------');
    disp('Run completed, final results:');
    disp(['h, then er, then mr, then ei, then mi.  method=',num2str(method)]);
    disp(['h ',h_str, '\\'])
    disp(['Error (real) & ',errorsr_str, '\\'])
    disp(['Order (real) & ',mr_str, '\\ \hline'])
    disp(['Error (imag) & ',errorsi_str, '\\'])
    disp(['Order (imag) & ',mi_str, '\\ \hline'])
    disp(['Error (mod2) & ',errorsm_str, '\\'])
    disp(['Order (mod2) & ',mm_str, '\\ \hline'])
    disp('------------------------');
    if(hi>1)
        mav = (mean(mr) + mean(mi))/2;
        disp(['m_av = ',num2str(mav)]);
    else
        mav = 0;
    end
    disp('--------------------------------------------------');
    if(length(hv)>1)
    if(method == 1 || method==11)
        strtmpm1 = ['CD:     m = ',num2str(mav,3)];     strtmp2 = '--sb';
    elseif(method == 2 || method ==12)
        strtmpm2 = ['2SHOC:  m = ',num2str(mav,3)];     strtmp2 = '-ok'; 
    end   
    
    fig_order = figure(fig_count); fig_count = fig_count+1;
    set(fig_order, 'Name','Method Orders','NumberTitle','off');
    loglog(hv,0.5*(errorsr+errorsi),strtmp2,'LineWidth',2)
    hold on;
    %loglog(hs,errorsi,strtmp2i,'LineWidth',2)
    %loglog(hs,errorsm,strtmp2m,'LineWidth',2)
    %legend('Real','Imag');    
    %a1 = annotation('textbox',[0.25 0.6 0.5 0.1]);
    %set(a1,'String',strtmp,'Fontsize',18,'FitBoxToText','on');    
    set(gca,'Fontsize',18);
    xlabel('h','Fontsize',18)
    set(gca,'XLim',[min(hv)-min(hv)/2, max(hv)+max(hv)/2]);
    set(gca,'XTick',fliplr(hv));
    set(gca,'XTickLabel',{'1/16','1/8','1/3','1/2','1'});
    ylabel('Ave Error Norm','Fontsize',18);
    end
end %IC=1

end %rmaxv

if(length(numframesv)>1)    
   numframetimes(ni)     = tcomp;
   numframechunksize(ni) = chunk_size;
end

end %numframesv

if(length(numframesv)>1)    
    if(exist('fig_frames','var')==0)
       fig_frames = figure(fig_count); fig_count = fig_count+1;
    end
    figure(fig_frames);      
    %Eliminate multiple times for same chunksize:
    [numframechunksize, m, n] = unique(numframechunksize);
    numframetimes = numframetimes(m);    
    set(fig_frames, 'Name','Chunksize versus Compute Time','NumberTitle','off');  
    if(precision==1)
        if(xreswanted == floor(1000.^(1/3)))
            mstr = 'r--o';
        elseif(xreswanted == floor((10000.^(1/3))))
            mstr = 'r--s';
        elseif(xreswanted == floor((100000.^(1/3))))
            mstr = 'r--^';
        elseif(xreswanted == floor((1000000.^(1/3))))
            mstr = 'r--d';
        end               
    else
        if(xreswanted == floor((1000.^(1/3))))
            mstr = 'k-o';
        elseif(xreswanted == floor((10000.^(1/3))))
            mstr = 'k-s';
        elseif(xreswanted == floor((100000.^(1/3))))
            mstr = 'k-^';
        elseif(xreswanted == floor((1000000.^(1/3))))
            mstr = 'k-d';
        end        
    end    
    plot(numframechunksize,numframetimes./min(numframetimes),mstr,'MarkerFaceColor','w','LineWidth',3);
    hold on;
    set(gca, 'XScale','log');
    set(gca,'Fontsize',20);
    axis([10 400 1 3]);
    %set(gca,'XTick',[10:10:200]);
    set(gca,'XTick',[10 25 50 100 200 400],'XTickLabel',[10 25 50 100 200 400]);  
    xlabel('Chunksize','Fontsize',20);
    ylabel('Slowdown Factor','Fontsize',20);        
end   
end %BC

if(track_ring==1 && length(rmaxv)>1)   
    %Plot Zvel:
    fig_zvel = figure(fig_count); fig_count = fig_count+1;
    for bi=1:length(BCv)
    BC = BCv(bi);    
    if(BC==2)
        pcl='k-'; pcl2 = 'ko';
    else
        pcl='b--'; pcl2 = 'bo';
    end   
    if(BC==2)            
     plot(rmaxv_msd,zvelCoM(:,bi),pcl,'LineWidth',3);    
    else  
     plot(rmaxv,    zvelCoM(:,bi),pcl,'LineWidth',3);     
    end
    hold on;    
    plot(rmaxv,c*ones(size(rmaxv)),'r-','LineWidth',2); 
    end
    plot(rmaxv,c*ones(size(rmaxv)),'r-','LineWidth',2);
    axis([min(rmaxv) max(rmaxv) min([zvelCoM(:);c-0.01]) max([zvelCoM(:);c+0.01])]);
    set(gca,'Fontsize',18);
    xlabel('r_p_a_d','Fontsize',18);
    ylabel('z-velocity','Fontsize',18);
    hold off;      
    
    %Plot Zvel versus c:  
    fig_zvelerr = figure(fig_count); fig_count = fig_count+1;
    for bi=1:length(BCv)
    BC = BCv(bi); 
    if(BC==2)
        pcl='k-'; pcl2 = 'ko';
    else
        pcl='b--'; pcl2 = 'bo';
    end   
    if(BC==2)  
     plot(rmaxv_msd,abs(zvelCoMerr(:,bi)),pcl,'LineWidth',3);     
     hold on;
    else  
     plot(rmaxv,abs(zvelCoMerr(:,bi)),pcl,'LineWidth',3);    
     hold on;
    end
    set(gca,'Fontsize',18);
    axis([min(rmaxv) max(rmaxv) min(zvelCoMerr(:)) max(zvelCoMerr(:))]);
    xlabel('r_p_a_d','Fontsize',18);
    ylabel('Percent Difference from c (%)','Fontsize',18);   
    end
    hold off;
    axis([min(rmaxv) max(rmaxv) min(abs(zvelCoMerr(:))) max(abs(zvelCoMerr(:)))]);    
    
  if(save_plots==1)   
  figure(fig_zvel)
  set(gcf, 'PaperPositionMode', 'auto');       
  print('-djpeg','-r100',['VRzvel_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(d),'.jpg'],['-f',num2str(fig_zvel)]); 
  print('-depsc','-r100',['VRzvel_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(d),'.eps'],['-f',num2str(fig_zvel)]); 
 
  figure(fig_zvelerr)
  set(gcf, 'PaperPositionMode', 'auto');       
  print('-djpeg','-r100',['VRzvelerr_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(d),'.jpg'],['-f',num2str(fig_zvelerr)]); 
  print('-depsc','-r100',['VRzvelerr_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(d),'.eps'],['-f',num2str(fig_zvelerr)]); 
 end  
end %track and rmaxv

if(length(rmaxv)>1)    
  if(length(BCv)==2)  
     fig_zvelbc = figure(fig_count); fig_count = fig_count+1;         
     %Assume MSD is first:
     zvelCoM(:,1)   = interp1(rmaxv_msd,zvelCoM(:,1),rmaxv);     
     zvelbcdiffcom  = 100*abs( 1 - (zvelCoM(:,2)./zvelCoM(:,1)));
     %zvelbcdiffgrid = 100*abs( 1 - (zvelgrid(:,2)./zvelgrid(:,1)));    
     plot(rmaxv,zvelbcdiffcom,'k-','LineWidth',3);
    % hold on;
     %plot(rmaxv,zvelbcdiffgrid,'ko','LineWidth',2);
     %hold off;
     set(gca,'Fontsize',fsize);
     axis([min(rmaxv) max(rmaxv) min(zvelbcdiffcom) max(zvelbcdiffcom)]);
     xlabel('r_p_a_d','Fontsize',18);
     ylabel('%-Diff of Z-Velocity MSD vs L0','Fontsize',18);   
     set(gcf, 'PaperPositionMode', 'auto');       
     print('-djpeg','-r100',['VRzvelBC_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(d),'.jpg'],['-f',num2str(fig_zvelbc)]); 
     print('-depsc','-r100',['VRzvelBC_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(d),'.eps'],['-f',num2str(fig_zvelbc)]); 
  end      
end %length(rmaxv)>!

end % Av

if(track_ring==1 && ((length(dv)>1 && IC==3) && (length(rmaxv)==1  && length(BCv)==1)))
  zvel_vs_dcom(di)  = abs(zvelCoM);
  zvel_vs_dgrid(di) = abs(zvelgrid);
  cv(di)            = abs(c);
  cv2(di)           = abs(c2); %used for leapfrog tests
end

end %dv

if(length(Av)>1)
    zvelAv'
end

if(track_ring==1 && ((length(dv)>1 && length(rmaxv)==1) && IC==3))    
     fig_zvel_d = figure(fig_count); fig_count = fig_count+1;     
     plot(dv,zvel_vs_dcom,'ko','LineWidth',3);
     hold on;
     %plot(dv,zvel_vs_dgrid,'ko','LineWidth',2);
     plot(dv,cv,'r--','LineWidth',2);
     if(num_vort==2)
     plot(dv,cv2,'b-','LineWidth',2);
     end
     hold off;
     set(gca,'Fontsize',18);
      axis([min(dv) max(dv)  min([zvel_vs_dcom(:);cv(:)]) max([zvel_vs_dcom(:);cv(:)])]);
     if(num_vort==2)
          axis([min(dv) max(dv)  min([zvel_vs_dcom(:);cv(:);cv2(:)]) max([zvel_vs_dcom(:);cv(:);;cv2(:)])]);         
     end  
     xlabel('d','Fontsize',18);
     ylabel('Z-Velocity','Fontsize',18);   
     if(save_plots==1) 
     set(gcf, 'PaperPositionMode', 'auto');       
     print('-djpeg','-r100',['VRzvel_d_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(dv(1)),'-',num2str(dv(end)),'_q',num2str(q),'.jpg'],['-f',num2str(fig_zvel_d)]); 
     print('-depsc','-r100',['VRzvel_d_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(dv(1)),'-',num2str(dv(end)),'_q',num2str(q),'.eps'],['-f',num2str(fig_zvel_d)]); 
     end
     fig_zvel_derr = figure(fig_count); fig_count = fig_count+1;     
     zvel_vs_dcomerr  = 100*(zvel_vs_dcom-cv)./cv;
     zvel_vs_dgriderr = 100*(zvel_vs_dgrid-cv)./cv;
     if(num_vort==2)
     zvel_vs_dcomerr2  = 100*(zvel_vs_dcom-cv2)./cv2;
     zvel_vs_dgriderr2 = 100*(zvel_vs_dgrid-cv2)./cv2;    
     end
     plot(dv,zvel_vs_dcomerr,'ko','LineWidth',3);
     if(num_vort==2)
     hold on;
     plot(dv,zvel_vs_dcomerr2,'bs','LineWidth',3);   
     hold off;
     end
    % plot(dv,zvel_vs_dgriderr,'ko','LineWidth',2);     
    % hold off;
     set(gca,'Fontsize',18);
     axis([min(dv) max(dv)  min(zvel_vs_dcomerr(:)) max(zvel_vs_dcomerr(:))]);     
     if(num_vort==2)
         axis([min(dv) max(dv)  min([zvel_vs_dcomerr(:);zvel_vs_dcomerr2(:)]) max([zvel_vs_dcomerr(:);zvel_vs_dcomerr2(:)])]);
     end     
     xlabel('d','Fontsize',18);
     ylabel('%-difference from c','Fontsize',18);   
     if(save_plots==1) 
     set(gcf, 'PaperPositionMode', 'auto');       
     print('-djpeg','-r100',['VRzvelerr_d_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(dv(1)),'-',num2str(dv(end)),'_q',num2str(q),'.jpg'],['-f',num2str(fig_zvel_derr)]); 
     print('-depsc','-r100',['VRzvelerr_d_T',num2str(endt),'_m',num2str(method),'_h',num2str(h),'_d',num2str(dv(1)),'-',num2str(dv(end)),'_q',num2str(q),'.eps'],['-f',num2str(fig_zvel_derr)]);   
     end
end

if((length(qv)>1 && IC==3) && (length(rmaxv)==1  && length(BCv)==1))
  zvel_vs_dcom(qi)  = abs(zvelCoM);
  zvel_vs_dgrid(qi) = abs(zvelgrid);
  cv(qi)            = abs(c);
  cv2(qi)           = abs(c2); %used for leapfrog tests  
  cv'
  cv2'
  zvel_vs_dcom'
end

end % qv

if(((length(cudav)>1)||length(methodv)==4) && length(xreswantedv)>1)
       fprintf(fid, '%7.3f\n', tcomp);  
end

end %xreswanted

if(((length(cudav)>1)||length(methodv)==4) && length(xreswantedv)>1)
   for ggg=1:length(xreswantedv)       
       fprintf(fid, '%7d\n',   xreswantedv(ggg));        
   end
   fclose(fid);
end

end % method

if(length(methodv)>1 && length(hv)>1)    
    legend(strtmpm1,strtmpm2)
    hold off;    
end

end %IC
end %precision

if(length(numframesv)>1)
   figure(fig_frames);
   hold off;
   if(save_plots==1) 
   set(gcf, 'PaperPositionMode', 'auto');       
   print('-djpeg','-r100',['3D_T',num2str(endt),'_N',num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),'_M',num2str(method),'.jpg'],['-f',num2str(fig_frames)]); 
   print('-depsc','-r100',['3D_T',num2str(endt),'_N',num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),'_M',num2str(method),'.eps'],['-f',num2str(fig_frames)]); 
   end
end
end %cuda

%Report wall time:
twall = toc(twall);
disp(['NLSE3D total Wall Time: ',num2str(twall),' seconds.']);

if(((length(cudav)>1)||length(methodv)==4) && length(xreswantedv)>1)
    diary off;
end

%Exit MATLAB is desired (good for nohup/profilers
if(exit_on_end==1),  exit; end;

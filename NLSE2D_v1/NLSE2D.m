%NLSE2D  Full Research Driver Script
%Program to integrate the two-dimensional Nonlinear Shr�dinger Equation:
%i*Ut + a*(Uxx+Uyy) - V(x,y)*U + s*|U|^2*U = 0.
%
%�2012 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center 
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com

%Clear any previous run variables and close all figures:
munlock; close all; clear all; clear classes; pause(0.5);
twall = tic; %Used to calculate total wall time.
%------------------Simulation parameters----------------------
endtw          = 100;     %Desireed end time of the simulation - may be slightly altered.
chunksizev     = 0;      %Run CUDA codes with different chunksizes.
numframesv     = [100];   %Number of times to view/plot/analyze solution.
hv             = [1/5];  %Spatial step-size.
%CD:
%klin  = 0.014142135623731
%kfull = 0.014072312614583
%2SHOC
%klin  = 0.010606601717798
%kfull = 0.010567277737245

k              = 0.01054;      %Time step-size.  Set to 0 to auto-compute largest stable size.
ICv            = [3];    %Select intial condition (1: linear guassian, 2: bright vortices, 3: dark vortices).
methodv        = [2];    %Spatial finite-difference scheme: 1:CD O(h^2), 2:2SHOC O(h^4), 11:Script CD, 12:Script 2SHOC.
runs           = 1;      %Number of times to run simulation.
xreswantedv    = 100;      %Desired grid size (N, N x N) (Overwrites IC auto grid-size calculations, and may be slightly altered)
cudav          = [1];    %Use CUDA integrators (0: no, 1: yes) 
precisionv     = [2];    %Use single (1) or double (2) precision integrators.
BCv            = [0];    %Overides boundary condition choice of IC (0:use IC BC 1:Dirichlet 2:MSD 3:Lap0.
rv             = 0;      %Used in MSD tests (Distance from center of grid to grid boundary).
tol            = 1.5;     %Modulus-squared tolerance to detect blowup.
%------------------Analysis switches----------------------
calc_mass      = 0;      %Calculates change in mass.
add_pert       = 0.0;    %Add uniform random perturbation of size e=add_pert.
track_vortices = 0;      %Track position of vortices.
%------------------Simulation switches--------------------
opt_1D         = 1;      %Optimize vortex radial profile using ODE for initial condition.
opt_2D         = 0;      %Optimize 2D IC as co-moving solution (used for a +1 -1 dark vortex senario).
pause_init     = 1;      %Pause initial condition before integation.
exit_on_end    = 0;      %Close MATLAB after simulation ends.
%------------------Plotting switches----------------------
show_waitbar   = 0;      %Show waitbar with completion time estimate.
plot_r_and_im  = 1;      %Plot modulus-squared as 3D surface along with real and imaginary parts as meshes.
plot_modulus2  = 1;      %Plot modulus-squared of solution (if above not set, as a 2D color plot).
plot_vorticity = 0;      %Plot vorticity of solution as 2D color plot.
vorticity_eps  = 0.01;  %Magnitude of added eps of denominator of vorticity (to avoid singularity).
plot_phase     = 0;      %Plot phase (arg(Psi)) of solution as 2D color plot.
%------------------Output switches----------------------
save_movies    = 0;   %Generate a gif (1) or avi (2) movie of simulation.
save_images    = 0;   %Saves images of each frame into eps (1) or jpg (2). 
disp_pt_data   = 0;   %Output value of Psi(5,5) at each frame.
%-----------------------------------------------------------------------------------
 
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
  cudafn1 =  [strcuda,'_2D_RES', num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),...
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

%Check for specific MSD test:    
if(rv==-1)
   rv2v  = 1;
else
   rv2v = 0;
end
    
bi = 0;
for BC_i=BCv
bi = bi+1;
ni = 0;
numframetimes     = zeros(size(numframesv)); %CUDA chunksize tests
numframechunksize = zeros(size(numframesv)); %CUDA chunksize tests
for numframes = numframesv    
ni = ni+1;  

%MSD tests, change rv depending on BC chosen:
if(rv2v==1)
  if(BC_i==3)        
         rv  = [18:1:35];
  else
         rv  = [12:1:35];
  end
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION PARAMETERS        %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
if(IC==1)  %Liniear EXP
    s  = 0;
    a  = 1;
    BC = 1;
    ymax =  sqrt(-2*a*log(sqrt(eps)));
    xmin = -ymax;
    xmax =  ymax;
    ymin = -ymax;
elseif(IC==2)   %Bright vortices
    a   = 1;
    s   = 1;
    amp = 0;
    BC  = 1;
    %Each vortices(:,i) places a vortex on the grid:
    %                m   OM   x0  y0  vx    vy     phase-shift
    vortices(:,1) = [1  0.15  0   0   0.0   0.0    0           ];      
elseif(IC==3)   %Dark vortices
    s        = -1;
    a        = 1;
    amp_mod2 = 1;              %Modulus-squared background amplitude.
    amp      = sqrt(amp_mod2); %Amplitude of initial Psi.
    BC       = 2;
    OM       = amp_mod2*s;    %Frequency (same for all vortices with equal background amp).
    %Each vortices(:,i) places a vortex on the grid:    
    %                m  om  x0  y0   vx  vy  phase-shift    
    vortices(:,1) = [1  OM  4   0    0.0 0.0 0          ];    
    vortices(:,2) = [1  OM  -4   0    0.0 0.0 0          ];
elseif(IC==4)   %Bright cubic-quintic vortices (not currently supported)
    a   = 1;
    s   = 1;
    amp = 0;
    BC  = 1;
    OMM = 0.155;    
     %Each vortices(:,i) places a vortex on the grid:  
    %                m  OM    x0   y0   vx     vy    phase-shift
    vortices(:,1) = [1  OMM   0    0    0.0    0.0   0          ];  
   %Only available cubic-quintic integrator is serial 2SHOC DP: 
   cuda=0;  method=2;  precision = 2;
   disp('Cubic-quintic NLSE selected, using serial MEX 2SHOC double precision...');
end

%If simulating vortices, automatically set grid to contain them properly:
%rmax1 allows for manual rmax values to be used.
if(IC>1)
    num_vort = length(vortices(1,:));
    ymax=0;ymin=0;xmax=0;xmin=0;
    for vi=1:length(vortices(1,:))
        if(IC==2) %Use VA ansatz to compute bounds:
            B     = sqrt((3*vortices(2,vi))/s);
            C     = sqrt((3*vortices(2,vi))/(2*a));
            R0    = sqrt((2*vortices(1,vi)^2*a)/vortices(2,vi));
            rmax  = (1/C)*asech((eps^(1/3))/B) + R0;
            rmax1 = rmax;           
        elseif(IC==3) %Use aysmptotic approx:
            m     = vortices(1,vi);
            OM    = vortices(2,vi);
            reps  = eps^(1/6);
            rmax1 = sqrt((-a*m^2)/(s*(2*reps*sqrt(OM/s) - reps^2)));
            rmax  = abs(m)*sqrt(-a/OM) + 20;%rmax1;%15;
            rmax1 = rmax;
        elseif(IC==4) %Use VA ansatz:
            OM       = vortices(2,vi)  ;
            OMstrtmp = rmc_bisect(@F_OM_str,0.01,0.1875,1e-15,OM);
            OMvatest = F_OM_va(OMstrtmp);
            OMtester = abs(OM-OMvatest);
            if(OMtester>1e-15)
                disp(['Warning, using vpa accuracy of ',num2str(200),'-digits for VA']);
                OMstrtmp = rmc_bisect(@F_OM_str,0.01,0.1875,vpa(1e-200),OM);
                OMvatest = F_OM_va(OMstrtmp);
                OMtester = abs(OM-OMvatest);
                if(double(OMtester)>1e-15)
                    disp('Warning! Could not get OM* within desired precision!');
                end
            end
            rmax = (1/(2*sqrt(OMstrtmp)))*acosh( (4*OMstrtmp - eps)/...
                   (eps*sqrt(1-(16/3)*OMstrtmp))) + vortices(1,vi)/sqrt(OMstrtmp-OM);
            rmax = double(rmax);
            OMstr(vi) = OMstrtmp;
            rmax1 = rmax;
        end %IC
        ymaxt = vortices(4,vi) + rmax1;
        xmaxt = vortices(3,vi) + rmax;
        ymint = vortices(4,vi) - rmax1;
        xmint = vortices(3,vi) - rmax;

        if(xmaxt>=xmax), xmax = xmaxt; end
        if(ymaxt>=ymax), ymax = ymaxt; end
        if(xmint<=xmin), xmin = xmint; end
        if(ymint<=ymin), ymin = ymint; end
    end
    %Manual overide of domain sizes:
    %ymin = -50
    %ymin = -rmax -(abs(m)/10)*endtw;
    %ymax = 35;  ymin=-ymax; xmax=ymax; xmin=ymin;
end %IC>1
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('----------------------------------------------------');
disp('--------------------NLSE2D--------------------------');
disp('----------------------------------------------------');
disp('Checking parameters...');

%Override BC from IC in case of specified BCs:
if(BCv(1)~=0)
   BC = BC_i;
end

%MSD test, make custom rv for MSD:
if(BC==2)
    rv_msd = rv;
end

%Start MSD test loop:
ri = 1;
for ri=1:length(rv)
   rmax = rv(ri);
   if(length(rv)>1)
      disp(['ri:  ',num2str(ri)]);
   end

%Set grid domain based on rmax:
if(rmax~=0)
 xmax = rmax;
 xmin = -rmax;
 ymax = rmax;
 ymin = -rmax;
end

%Step size loop (for scheme order analysis)
hi=0;
for h = hv
hi = hi+1;

%Adjust ranges so they are in units of h (fixes bugs):
xmin = xmin - rem(xmin,h); ymin = ymin - rem(ymin,h);
xmax = xmax - rem(xmax,h); ymax = ymax - rem(ymax,h);
%Update rv accordingly
rv(ri) = rv(ri)-max(abs([rem(xmin,h), rem(xmax,h),...
                         rem(ymin,h), rem(ymax,h)]));
%Update original MSD rv as well:
if(BC==2)
    rv_msd(ri) = rv(ri);
end

%Overide limits for CUDA timings:
if(xreswanted>0)
    xmax =  (xreswanted-1)*h/2;    xmin = -(xreswanted-1)*h/2;
    ymax =  (xreswanted-1)*h/2;    ymin = -(xreswanted-1)*h/2;
end

%Set up spatial grid:
xvec  = xmin:h:xmax; xres = length(xvec);
yvec  = ymin:h:ymax; yres = length(yvec);

%Need meshgrid matrices to make ICs:
[X,Y] = meshgrid(xvec,yvec);
[M N] = size(X);

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
        cudablocksizex = 16;            
        cudablocksizey = 16;      
               
        sharedmemperblock = (cudablocksizex+2)*(cudablocksizey+2)*(3+2*method)*(4*precision)/1000;
        numcudablocksx = ceil(M/cudablocksizex);    
        numcudablocksy = ceil(N/cudablocksizey);
        numcudablocks  = numcudablocksx*numcudablocksy;
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
            
            if(msdalterflag==1)
               %Have to update rmax vector for msd tests for accurate results
               rv_msd(ri) = rv(ri)-h;
            end  
            
            %Now, recompute grid:
            xvec  = xmin:h:xmax;  xres  = length(xvec);
            yvec  = ymin:h:ymax;  yres  = length(yvec);
            [X,Y] = meshgrid(xvec,yvec);    
            
            %Reset numblocks with new gridsize:
            [M N] = size(X);
            numcudablocksx = ceil(M/cudablocksizex);    
            numcudablocksy = ceil(N/cudablocksizey);
            numcudablocks  = numcudablocksx*numcudablocksy;
        end       
    else
        disp('Sorry, it seems CUDA is not installed');
        cuda=0;
    end
end%cuda1

for run_i=1:runs %Run loop (for averaging timings)

%Initialize solution and potential matrices:
U     = zeros(size(X));
V     = zeros(size(X));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION FORMULATION       %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
disp('----------------------------------------------------');
disp('Setting up initial condition...---------------------');
%Formulate Initial Condition:
if(IC==2||IC==3) %NLS Vortices:
    %Initialize U to be background amplitude:
    U = U + amp;
    %Add vortices:
    for vi=1:num_vort
        U = NLS_add_vortex(vortices(1,vi),vortices(2,vi),vortices(3,vi),...
                           vortices(4,vi),vortices(5,vi),vortices(6,vi),...
                           vortices(7,vi),U,h,a,s,amp,X,Y,opt_1D);
    end
%Two-dimensional co-moving optimization for 2 opposite charge dark
%vortices:
if(opt_2D==1)
    sizeu = size(U);
    M = sizeu(1);
    N = sizeu(2);
    c = -1/d;  %need to define d as distance of vortices/2 above
    par = [OM a s h M N xmin xmax ymin ymax];
    maxit         = 20;
    maxitl        = 20;
    etamax        = 0.7;
    lmeth         = 3;
    restart_limit = 10;
    sol_parms=[maxit,maxitl,etamax,lmeth,restart_limit];
    
    calc_c=0;
    %Now refined 2D solution using 2D nsoli:
    if(calc_c==0)
       U = U.*exp(-1i.*Y.*(c/(2*a))); 
       U = [real(U(:)) ; imag(U(:))];
       [U,it_hist,ierr] = nsoli(U,  @(Utmp)NLSE2D_STEADY_F(Utmp,par),  1e-8*[1,1],sol_parms);
       ierr
       rhs = NLSE2D_STEADY_F(U,par);       
    else
       U = [real(U(:)) ; imag(U(:)); c]; 
       [U,it_hist,ierr] = nsoli(U,  @(Utmp)NLSE2D_STEADY_Fc(Utmp,par),  1e-8*[1,1],sol_parms);
       ierr            
       rhs = NLSE2D_STEADY_Fc(U,par);
       rhs = squeeze(rhs(1:end-1));
       c = U(end)     
       U = squeeze(U(1:end-1));
    end    
    rhs = rhs(1:(length(rhs))/2) + 1i*rhs(((length(rhs))/2 + 1):end);
    rhs = reshape(rhs,M,N);
    figure(10); surf(X,Y,abs(rhs).^2); xlabel('x');ylabel('y');zlabel('RHS'); shading interp;    axis tight;
    U = U(1:(length(U))/2) + 1i*U(((length(U))/2 + 1):end);
    U = reshape(U,M,N);
    U = U.*exp(1i.*Y.*(c/(2*a)));      
end%opt_2D
elseif(IC==4) %Cubic-quintic Vortices
    for vi=1:num_vort
        U = CQNLS_add_vortex(vortices(1,vi),vortices(2,vi),OMstr(vi),vortices(3,vi),...
                           vortices(4,vi),vortices(5,vi),vortices(6,vi),...
                           vortices(7,vi),U,h,a,s,amp,X,Y);
    end
elseif(IC==1) %Linear steady-state exp profile
    V    = (X.^2+Y.^2)/a;
    U    = exp(-(X.^2+Y.^2)/(2*a));
    Ureal = U;
end

%Save initial condition for MSD error comparisons:
if(length(rv)>1)
    U2true = U.*conj(U);
end

%Add random perturbation if specified:
if(add_pert>0)
    U = U.*(1 + add_pert.*rand(size(U)));
end

%Check initial condition for problems:
if(max(real(U(:))) > tol || sum(isnan(U(:))) > 0)
   disp('Initial Condition Has a PROBLEM'); 
   break;
end

%-------------------------------------------------------------------------%
%Compute stability bounds for time step k:
if(k==0)  
   hmin = min(hv);  
   if(method==1 || method==11)  
      G = [8,7,6,2,1,0]; 
      klin = hmin^2/(2*sqrt(2)*a);
   elseif((method==2 || method==12) || length(methodv)>1)
      G = (1/12)*[128,127,126,110,109,92,24,9,8,-6,-7,-8];
      klin = (3/4)*hmin^2/(2*sqrt(2)*a);
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
       ub1   = U(2,2:end-1);     ub2 = U(2:end-1,2);     
       ub3   = U(end-1,2:end-1); ub4 = U(2:end-1,end-1); 
       UB_1  = [ub1(:); ub2(:); ub3(:); ub4(:)];
       Ut    = NLSE2D_F(U,V,hmin,s,a,method,BC,(1/hmin^2),(a/12),(a/(6*hmin^2)),(1/a));
       utb1  = Ut(2,2:end-1);     utb2 = Ut(2:end-1,2); 
       utb3  = Ut(end-1,2:end-1); utb4 = Ut(2:end-1,end-1); 
       UtB_1 = [utb1(:); utb2(:); utb3(:); utb4(:)];                     
       Bmax = real(max((hmin^2/(1i*a))*UtB_1./UB_1));
       clear UtB_1 UB_1 Ut ub1 ub2 ub3 ub4 utb1 utb2 utb3 utb4;
    elseif(BC==3)
       u1 = U(1,:);   u2 = U(:,1);
       u3 = U(end,:); u4 = U(:,end);
       UB = [u1(:); u2(:); u3(:); u4(:)];           
       v1 = V(1,:);   v2 = V(:,1);
       v3 = V(end,:); v4 = V(:,end); 
       VB = [v1(:); v2(:); v3(:); v4(:)];  
       Bmax = max((hmin^2/a)*(s*UB.*conj(UB) - VB));
       clear VB UB u1 u2 u3 u4 v1 v2 v3 v4;
   end    
   %Now compute full k bound:
   kfull = (hmin^2/a)*sqrt(8)/max(abs(Bmax),max(abs(Lmaxes)));       
   disp(['kmax (linear): ',num2str(klin)]);
   disp(['kmax   (full): ',num2str(kfull)]);
   if(kfull<klin)
      k = 0.9*kfull;
   else
      k = 0.9*klin; 
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
disp('2D NLSE Parameters:')
disp(['a:          ',num2str(a)]);
disp(['s:          ',num2str(s)]);
disp(['xmin:       ',num2str(xmin)]);
disp(['xmax:       ',num2str(xmax)]);
disp(['ymin:       ',num2str(ymin)]);
disp(['ymax:       ',num2str(ymax)]);
disp(['Endtime:    ',num2str(endt)]);
disp('Numerical Parameters:');
disp(['h:             ',num2str(h)]);
disp(['k:             ',num2str(k)]);
disp(['GridSize(x,y): ',num2str(xres),'x',num2str(yres), ' = ',num2str(xres*yres)]);
disp(['TimeSteps:     ',num2str(steps)]);
disp(['ChunkSize:     ',num2str(chunk_size)]);
disp(['NumFrames:     ',num2str(numframes)]);
if(precision==1)
disp('Precision:      Single');
else
disp('Precision:      Double');
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
    disp('Boundary Conditions:  MSD |U|^2 = B');
elseif(BC==3)
    disp('Boundary Conditions:  Uxx+Uyy = 0');
elseif(BC==4)
    disp('Boundary Conditions:  One-Sided Diff');
end
if(cuda == 1)
    disp( 'CUDA Parameters and Info:');
    disp(['BlockSize:     ',num2str(cudablocksizex),'x',num2str(cudablocksizey)]);
    disp(['CUDAGridSize:  ',num2str(numcudablocksx),'x',num2str(numcudablocksy)]);
    disp(['NumBlocks:     ',num2str(numcudablocks)]); 
    disp(['Shared Memory/Block: ',num2str(sharedmemperblock),'KB']);
    disp(['TotalGPUMemReq:      ',num2str(yres*xres*(9 + 2*(method-1))*(4*precision)/1024),'KB']); 
end
if(save_movies==1)
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
%%%%%%%%%   

maxMod2U  = max(U(:).*conj(U(:)));
maxRealU  = max(real(U(:)));
plot_max  = max([maxMod2U maxRealU]);
fsize     = 18;  %Set font size for figures.
fig_count = 1;

if(plot_vorticity==1 || track_vortices==1)
    %Compute Vorticity:            
    [grad_x,grad_y] = gradient(U,h,h);
    V_x = (conj(U).*grad_x - U.*conj(grad_x) ) ./(1i*(2*U.*conj(U)+vorticity_eps));
    V_y = (conj(U).*grad_y - U.*conj(grad_y) ) ./(1i*(2*U.*conj(U)+vorticity_eps));
    [Vxx,Vxy]=gradient(V_x,h,h);
    [Vyx,Vyy]=gradient(V_y,h,h);
    W = Vyx - Vxy;    
end

if(plot_modulus2==1)

fig_mod2  = figure(fig_count);
fig_count = fig_count+1;
set(fig_mod2, 'Name','2D NLSE MOD2 t = 0','NumberTitle','off','Color','w');

%Plot mod-squared initial condition:
plot_mod2  = surf(X,Y,U.*conj(U), 'EdgeColor', 'none', 'EdgeAlpha', 1);
shading interp;
%colormap(flipud(jet(512)));
colormap(hsv(512));
%colormap(bone(500));
axis([xmin xmax ymin ymax -0.01 (maxMod2U+maxMod2U)]);
set(gca,'DataAspectRatio',[1 1 2*(2*maxMod2U)/(xmax-xmin)]);
xlabel('x','FontSize',fsize); ylabel('y','FontSize',fsize); zlabel('\Psi','FontSize',fsize);
set(gca,'Fontsize',fsize);
caxis([-0.01 maxMod2U+0.1*maxMod2U]);
view([0 90]);
%3D surface plot:
if(plot_r_and_im==1)
    axis([xmin xmax ymin ymax -(plot_max+0.1*plot_max) (plot_max+0.1*plot_max)]);
    colormap(bone(500));
    %colormap(flipud(jet(512)));
    hold on;
    plot_real  = surf(X,Y,real(U));
    plot_imag  = surf(X,Y,imag(U));
    hold off;
    set(plot_real, 'FaceColor', 'none','Meshstyle','row',   ...
      'EdgeColor', 'b'  , 'EdgeAlpha', 0.1,'LineWidth',1);
    set(plot_imag, 'FaceColor', 'none','Meshstyle','column',...
      'EdgeColor', 'r'  , 'EdgeAlpha', 0.1,'LineWidth',1);
    axis vis3d;
    view([-17 10]);
end
drawnow;
end %plotmod2

%time_txt   = ['Time:      0.00'];
%plot_label = text((1/4)*xmax,(3/4)*ymax,1.25*maxMod2U,time_txt,'BackgroundColor','white','FontSize',fsize/2);

%Plot vorticity:
if(plot_vorticity==1)
    fig_vort  = figure(fig_count);
    fig_count = fig_count+1;
    pfig_vort = surf(X,Y,W, 'EdgeColor', 'none', 'EdgeAlpha', 1);
    set(fig_vort, 'Name','2D NLSE Vorticity  t = 0','NumberTitle','off','Color','w');
    shading interp;
    grid off
    xlabel('x','FontSize',fsize);
    ylabel('y','FontSize',fsize);
    set(gca,'FontSize',fsize);
    plotw_max = max(W(:));
    plotw_min = min(W(:));
    dratio = 2*plotw_max / (xmax-xmin) ;
    set(gca,'DataAspectRatio',[1 1 2*dratio]);
    caxis([plotw_min plotw_max]);
    axis([xmin xmax ymin ymax (plotw_min + 0.1*plotw_min) (plotw_max+0.1*plotw_max)]);
    axis 'auto z';
    colormap(jet(2000))
    view([0 90]);
    drawnow;
end
%Plot phase:
if(plot_phase==1)
    fig_phase  = figure(fig_count);
    fig_count  = fig_count+1;
    pfig_phase = surf(X,Y,angle(U), 'EdgeColor', 'none', 'EdgeAlpha', 1);
    set(fig_phase, 'Name','Phase  t = 0','NumberTitle','off','Color','w');
    shading interp;
    grid off
    xlabel('x','FontSize',fsize);
    ylabel('y','FontSize',fsize);
    set(gca,'FontSize',fsize);
    plotp_max = max(angle(U(:)));
    plotp_min = min(angle(U(:)));
    dratio = 2*plotp_max / (xmax-xmin) ;
    set(gca,'DataAspectRatio',[1 1 2*dratio]);
    caxis([plotp_min plotp_max]);
    axis([xmin xmax ymin ymax (plotp_min + 0.1*plotp_min) (plotp_max+0.1*plotp_max)]);
    axis 'auto z';
    colormap(jet(2000))
    view([0 90]);
    drawnow;
end

%Setup filenames for movies/images:
if(save_movies>=1 || save_images>=1)
if(IC==1)
    fn = 'linear';
elseif(IC==2 || IC==3)  
    sizetmp = size(vortices);
    fn = ['nlse2D_',num2str(sizetmp(2)),'-vort_s',num2str(s),'_m' num2str(vortices(1,1)),'_h=',num2str(h),'_k=',num2str(k),'_endt=',num2str(endt)]; 
    fn = strrep(fn,'.','');    
elseif(IC==4) 
    sizetmp = size(vortices);
    fn = ['cqnls_m',num2str(vortices(1,1)),'_om',num2str(vortices(2,1)),'_dxy',num2str(h),'_t',num2str(endt)];
    if sizetmp(2)==2
        %m  om     x0   y0   vx      vy     phase
        fn = ['cqnls_om',num2str(vortices(2,1)),'_v1_m',num2str(vortices(1,1)),'_p',num2str(vortices(7,1)),...
          '_v2_m',num2str(vortices(1,2)),'_p',num2str(vortices(7,2)),...
          '_vel',num2str(vortices(5,1)),'_dxy',num2str(h),'_endt',num2str(endt),'_dt',num2str(k)];
    end
end
end

%Setup waitbar:
if(show_waitbar==1)
    fig_waitbar = waitbar(0,'Estimated time to completion ???:??');
    set(fig_waitbar, 'Name','Simulation 0% Complete','NumberTitle','off');
    waitbar(0,fig_waitbar,'Estimated time to completion ???:??');
end

%Save images (saves initial condition before pause)
if(save_images==1)
    if(plot_modulus2==1) 
       set(fig_mod2, 'PaperPositionMode', 'auto'); 
       if(plot_r_and_im==1)           
           print('-depsc','-r125','-opengl',[fn '_MOD2_t=0'],['-f',num2str(fig_mod2)]); 
       else
           print('-depsc','-r125','-opengl',[fn '_MOD2_t=0'],['-f',num2str(fig_mod2)]); 
       end
    end
    if(plot_vorticity==1)    
       set(fig_vort, 'PaperPositionMode', 'auto'); 
       print('-depsc','-r125','-opengl',[fn '_VORT_t=0'],['-f',num2str(fig_vort)]);   
    end
    if(plot_phase==1)       
      set(fig_phase, 'PaperPositionMode', 'auto'); 
      print('-depsc','-r125','-opengl',[fn '_PHASE_t=0'],['-f',num2str(fig_phase)]);   
    end       
end
if(save_images==2)
    if(plot_modulus2==1) 
       set(fig_mod2, 'PaperPositionMode', 'auto'); 
       print('-djpeg','-r125','-opengl',[fn '_MOD2_t=0.jpg'],['-f',num2str(fig_mod2)]); 
    end
    if(plot_vorticity==1)       
       set(fig_vort, 'PaperPositionMode', 'auto'); 
       print('-djpeg','-r125','-opengl',[fn '_VORT_t=0.jpg'],['-f',num2str(fig_vort)]);   
    end
    if(plot_phase==1)     
      set(fig_phase, 'PaperPositionMode', 'auto'); 
      print('-djpeg','-r125','-opengl',[fn '_PHASE_t=0.jpg'],['-f',num2str(fig_phase)]);   
    end       
end

%Pause simulation if desired.
if(pause_init==1)
    disp('Initial condition displayed.  Press a key to start simulation');
    pause;
end

%Setup movie files:
if(save_movies==1)
    if(plot_modulus2==1)        
       gifname = [fn,'_MOD2.gif'];
       I = getframe(fig_mod2);
       I = frame2im(I);
       [XX, map] = rgb2ind(I, 128);
       imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
       'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
    end
    if(plot_vorticity==1)        
       gifname = [fn,'_VORT.gif'];
       I = getframe(fig_vort);
       I = frame2im(I);
       [XX, map] = rgb2ind(I, 128);
       imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
       'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
    end
    if(plot_phase==1)        
       gifname = [fn,'_PHASE.gif'];
       I = getframe(fig_phase);
       I = frame2im(I);
       [XX, map] = rgb2ind(I, 128);
       imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
          'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
    end    
elseif(save_movies==2)
    if(plot_modulus2==1)
        mov_mod2 = avifile([fn,'_MOD2.avi'],'compression','None','quality',100);    
        mov_mod2 = addframe(mov_mod2,getframe(fig_mod2)); 
    end
    if(plot_phase==1)
        mov_phase = avifile([fn,'_PHASE.avi'],'compression','None','quality',100);    
        mov_phase = addframe(mov_phase,getframe(fig_phase)); 
    end
    if(plot_vorticity==1)
        mov_vort = avifile([fn,'_VORT.avi'],'compression','None','quality',100);    
        mov_vort = addframe(mov_vort,getframe(fig_vort)); 
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

%Setup tracking variables:
if(track_vortices==1)
   xv       = zeros(num_vort,numframes);
   yv       = zeros(num_vort,numframes);
   xiv      = zeros(num_vort,numframes);
   yiv      = zeros(num_vort,numframes);
   v_trackY = zeros(num_vort,numframes);
   v_trackX = zeros(num_vort,numframes);  
end

plottime  = zeros(numframes,1);
if(IC==1) %Setup error vectors for linear test:
   errorveci = zeros(numframes,1);
   errorvecr = zeros(numframes,1);
   errorvecm = zeros(numframes,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             BEGIN SIMULATION                    %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
crash=0;
for chunk_count = 1:numframes
   if(crash==1), break; end
   %Start chunk-time timer (includes plots)
   if(show_waitbar==1), tchunk  = tic; end
   tcompchunk = tic;  %Start compute-time timer
   
   %Call MEX integrator codes:
   if(method==1)
    if(precision==1)
        if(cuda==0)
            U = NLSE2D_TAKE_STEPS_CD_F(U,V,s,a,h^2,BC,chunk_size,k);
        else
            U = NLSE2D_TAKE_STEPS_CD_CUDA_F(U,V,s,a,h^2,BC,chunk_size,k);
        end
    elseif(precision==2)
        if(cuda==0)
            U = NLSE2D_TAKE_STEPS_CD(U,V,s,a,h^2,BC,chunk_size,k);
        else
            U = NLSE2D_TAKE_STEPS_CD_CUDA_D(U,V,s,a,h^2,BC,chunk_size,k);
        end
    end
   elseif(method==2)

     if(precision==1)
        if(cuda==0)
            U = NLSE2D_TAKE_STEPS_2SHOC_F(U,V,s,a,h^2,BC,chunk_size,k);
        else
            U = NLSE2D_TAKE_STEPS_2SHOC_CUDA_F(U,V,s,a,h^2,BC,chunk_size,k);
        end
    elseif(precision==2)
        if(cuda==0)
            if(IC==4) %Use cubic-quintic integrator:
               U = CQNLSE2D_TAKE_STEPS_2SHOC(U,V,1,-1,a,h^2,BC,chunk_size,k);
            else
               U = NLSE2D_TAKE_STEPS_2SHOC(U,V,s,a,h^2,BC,chunk_size,k);
            end
        else
            U = NLSE2D_TAKE_STEPS_2SHOC_CUDA_D(U,V,s,a,h^2,BC,chunk_size,k);
        end
     end
   else  %Old code for trial methods / mex timings
    %Do divisions first to save compute-time:
    k2    = k/2;    k6    = k/6;    l_a   = 1/a;
    l_h2  = 1/h^2;    a_12  = a/12;    a_6h2 = a/(6*h^2);

    for nc = 1:chunk_size
       %Start Runga-Kutta:
       k_tot   = NLSE2D_F(U,V,h,s,a,method,BC,l_a,l_h2,a_12,a_6h2); %K1
       %----------------------------
       Uk_tmp  = U + k2*k_tot;
       Uk_tmp  = NLSE2D_F(Uk_tmp,V,h,s,a,method,BC,l_a,l_h2,a_12,a_6h2); %K2
       k_tot   = k_tot + 2*Uk_tmp; %K1 + 2K2
       %----------------------------
       Uk_tmp  = U + k2*Uk_tmp;
       Uk_tmp  = NLSE2D_F(Uk_tmp,V,h,s,a,method,BC,l_a,l_h2,a_12,a_6h2); %K3
       k_tot   = k_tot + 2*Uk_tmp; %K1 + 2K2 + 2K3
       %----------------------------
       Uk_tmp  = U + k*Uk_tmp;
       k_tot   = k_tot + NLSE2D_F(Uk_tmp,V,h,s,a,method,BC,l_a,l_h2,a_12,a_6h2); %K1 + 2K2 + 2K3 + K4
       %-------------------------------
       U = U + k6*k_tot; %New time step
    end %chunksize
   end%method step

   tcomp = tcomp + toc(tcompchunk); %Add comp-chunk time to compute-time
   n = n + chunk_size;

  %Detect blow-up:
  
  if(max(abs(U(:))) > tol || sum(isnan(U(:))) > 0)
     disp('CRAAAAAAAAAAAAAAAASSSSSSSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHH');
     disp(['CRASH!  h=',num2str(h),' k=',num2str(k),' t=',num2str(n*k,'%.2f')]);  
  if( sum(isnan(U(:))) > 0)
     disp('NAN Found!');
  else
     disp('Tolerance Exceeded!');
  end  
  disp('CRAAAAAAAAAAAAAAAASSSSSSSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHH');
  crash=1;
  %break;
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%PLOT%%%SOLUTION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(plot_vorticity==1 || track_vortices==1)   
    %Compute Vorticity:
    [grad_x,grad_y] = gradient(U,h,h);
    V_x = (conj(U).*grad_x - U.*conj(grad_x) ) ./(1i*(2*U.*conj(U)+vorticity_eps));
    V_y = (conj(U).*grad_y - U.*conj(grad_y) ) ./(1i*(2*U.*conj(U)+vorticity_eps));
    [Vxx,Vxy]=gradient(V_x,h,h);
    [Vyx,Vyy]=gradient(V_y,h,h);
    W = Vyx - Vxy;    
end

if(plot_modulus2==1)
   set(fig_mod2, 'Name',['2D NLSE MOD2 t = ',num2str((n*k),'%.2f')]);
   set (plot_mod2, 'Zdata', U.*conj(U));
   if(plot_r_and_im==1)
     set (plot_real, 'Zdata', real(U));
     set (plot_imag, 'Zdata', imag(U));
   end
   %time_txt   = ['Time:      ',num2str((n*k),'%.2f')];
   %set(plot_label,'String',time_txt);
   if(save_movies==1)
       gifname = [fn,'_MOD2.gif'];
       I = getframe(fig_mod2);
       I = frame2im(I);
       [XX, map] = rgb2ind(I, 128);
       imwrite(XX, map, gifname, 'GIF', 'WriteMode',...
                'append', 'DelayTime', 0);
   elseif(save_movies==2)
      mov_mod2 = addframe(mov_mod2,getframe(fig_mod2));
   end  
end%plotmod2

if(plot_vorticity==1)   
    set(pfig_vort, 'Zdata', W);
    set( fig_vort, 'Name',['NLSE2D Vorticity  t = ',num2str((n*k),'%.2f')]);
    if(save_movies==1)
       gifname = [fn,'_VORT.gif'];
       I = getframe(fig_vort);
       I = frame2im(I);
       [XX, map] = rgb2ind(I, 128);
       imwrite(XX, map, gifname, 'GIF', 'WriteMode',...
                'append', 'DelayTime', 0);
    elseif(save_movies==2)
      mov_vort = addframe(mov_vort,getframe(fig_vort));
    end    
end
if(plot_phase==1)
    set(pfig_phase, 'Zdata', angle(U));
    set( fig_phase, 'Name',['Phase  t = ',num2str((n*k),'%.2f')]);
    if(save_movies==1)
       gifname = [fn,'_PHASE.gif'];
       I = getframe(fig_phase);
       I = frame2im(I);
       [XX, map] = rgb2ind(I, 128);
       imwrite(XX, map, gifname, 'GIF', 'WriteMode',...
                'append', 'DelayTime', 0);
    elseif(save_movies==2)
      mov_phase = addframe(mov_phase,getframe(fig_phase));
    end
end%plotphase

if(save_images==1)
    if(plot_modulus2==1) 
       set(fig_mod2, 'PaperPositionMode', 'auto'); 
       print('-depsc','-r125','-opengl',[fn '_MOD2_t=' num2str(n*k) '.eps'],['-f',num2str(fig_mod2)]);   
    end
    if(plot_vorticity==1)        
       set(fig_vort, 'PaperPositionMode', 'auto'); 
       print('-depsc','-r125','-opengl',[fn '_VORT_t=' num2str(n*k) '.eps'],['-f',num2str(fig_vort)]);   
    end
    if(plot_phase==1)        
      set(fig_phase, 'PaperPositionMode', 'auto'); 
      print('-depsc','-r125','-opengl',[fn '_PHASE_t=' num2str(n*k) '.eps'],['-f',num2str(fig_phase)]);   
    end       
end
if(save_images==2)
    if(plot_modulus2==1) 
       set(fig_mod2, 'PaperPositionMode', 'auto'); 
       print('-djpeg','-r125','-opengl',[fn '_MOD2_t=' num2str(n*k) '.jpg'],['-f',num2str(fig_mod2)]);   
    end
    if(plot_vorticity==1)        
       set(fig_vort, 'PaperPositionMode', 'auto'); 
       print('-djpeg','-r125','-opengl',[fn '_VORT_t=' num2str(n*k) '.jpg'],['-f',num2str(fig_vort)]);   
    end
    if(plot_phase==1)        
       set(fig_phase, 'PaperPositionMode', 'auto'); 
      print('-djpeg','-r125','-opengl',[fn '_PHASE_t=' num2str(n*k) '.jpg'],['-f',num2str(fig_phase)]);   
    end       
end

%Refresh plots if no movie saved (movie auto-refreshes)
if(save_movies==0)
    drawnow;
end

%Display single point for error comparason/convergance tests: 
if(disp_pt_data==1)
    disp(['2D NLSE  t = ',num2str((n*k),'%.2f'),'U(5,5) = ',num2str(U(5,5))]);
end

%%%Analysis Tools%%%%%%%%%*********************************************
calcn           = calcn+1;
plottime(calcn) = n*k;

%Calculate mass:
if(calc_mass==1)
   mass(calcn) = sum(U(:).*conj(U(:))).*h^2;
   delta_mass = (abs(mass(1)-mass(calcn))/mass(1))*100;
end

%Compute MSD error:
if(length(rv)>1)
    errorvecm(calcn) = max( abs(U2true(:)  -  U(:).*conj(U(:))) );
end

%Record error for linear test (IC=1)
if(IC==1)
    Ureal = exp(-(X.^2+Y.^2)/(2*a)).*exp(-1i*2*k*n);
    errorvecr(calcn) = sqrt(sum((real(Ureal(:)) - real(U(:))).^2)/length(U(:)));
    errorveci(calcn) = sqrt(sum((imag(Ureal(:)) - imag(U(:))).^2)/length(U(:)));
    errorvecm(calcn) = sqrt(sum((abs(Ureal(:)).^2  -  abs(U(:)).^2).^2)/length(U(:)));
end

%Track dark vortices:
if(IC==3 && track_vortices==1)
    Yvec = squeeze(Y(:,1));
    Xvec = squeeze(X(1,:));    
    sizeu = size(W); 
    %Set radius to scan for updated vortex position:
    scan_radius = 2*sqrt(abs(a/OM));
    sr_i = floor(scan_radius/h);
    
    for vi=1:num_vort    
       if(calcn==1)   
          %Get initial positions of vortices:   
           yv(vi,calcn) = vortices(4,vi);
           xv(vi,calcn) = vortices(3,vi);
           [tmp,yiv(vi,calcn)] = min(abs(Yvec-yv(vi,calcn)));
           [tmp,xiv(vi,calcn)] = min(abs(Xvec-xv(vi,calcn)));           
           yv(vi,calcn) = Yvec(yiv(vi,calcn));         
           xv(vi,calcn) = Xvec(xiv(vi,calcn));  
           yivmin = yiv(vi,calcn)-sr_i;  
           xivmin = xiv(vi,calcn)-sr_i;    
           yivmax = yiv(vi,calcn)+sr_i;  
           xivmax = xiv(vi,calcn)+sr_i;
       else     
       %Set scan-box:
       yivmin = yiv(vi,calcn-1)-sr_i;  
       xivmin = xiv(vi,calcn-1)-sr_i;    
       yivmax = yiv(vi,calcn-1)+sr_i;  
       xivmax = xiv(vi,calcn-1)+sr_i;    
       %Find new index in box of location of center of vortex:
       [yiv(vi,calcn), xiv(vi,calcn)] = ...
           find(    abs(W(yivmin:yivmax,xivmin:xivmax)) == ...
            max(max(abs(W(yivmin:yivmax,xivmin:xivmax)))));
       
       %Compute index location of vortex position: 
       yiv(vi,calcn) = yivmin + yiv(vi,calcn) - 1;
       xiv(vi,calcn) = xivmin + xiv(vi,calcn) - 1;   
       yv(vi,calcn)  = Yvec(yiv(vi,calcn));
       xv(vi,calcn)  = Xvec(xiv(vi,calcn));    
       end     
       
       %Compute "center of mass" location of vortex around index position:
       Mv = sum(sum(abs(W(yivmin:yivmax,xivmin:xivmax)))*h^2);
       v_trackY(vi,calcn)  = h^2*sum(sum(Y(yivmin:yivmax,xivmin:xivmax).* ...
                                     abs(W(yivmin:yivmax,xivmin:xivmax))))/Mv;       
       v_trackX(vi,calcn)  = h^2*sum(sum(X(yivmin:yivmax,xivmin:xivmax).* ...
                                     abs(W(yivmin:yivmax,xivmin:xivmax))))/Mv;  
    end %vi
end %Trackvortices

%Update waitbar:
if(show_waitbar==1)
    time4frame = toc(tchunk);
    totsecremaining = (numframes-chunk_count)*time4frame;
    minremaining = floor(totsecremaining/60);
    secremaining = floor(60*(totsecremaining/60 - floor(totsecremaining/60)));
    set(fig_waitbar, 'Name',['Simulation ',num2str((n/steps)*100),'% Complete']);
    waitbar(n/steps,fig_waitbar,['Estimated time to completion ',num2str(minremaining,'%03d'),...
    ':',num2str(secremaining,'%02d')]);
end

end %chunk-count
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%         SIMULATION IS OVER                      %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Clear non-mex RK4 matrices if used:
if(method>10)
   clear Uk_tmp;
   clear k_tot;
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
if(save_movies==2)
   if(plot_modulus2==1),  mov_mod2  = close(mov_mod2);  end;
   if(plot_phase==1),     mov_phase = close(mov_phase); end;
   if(plot_vorticity==1), mov_vort  = close(mov_vort);  end;        
end
end %Repeated runs complete.

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
   title_txt = ['\DeltaMass = ',num2str(delta_mass,'%.4f'),'%','\newline', ...
              title_txt_params];
   title(title_txt);
   grid on;
   xlabel('Time');   ylabel('Mass');
end

%Plot MSD errors:
if(length(rv)>1)
    errorsm = max(errorvecm);
    fig_err = figure(fig_count);
    fig_count = fig_count + 1;
    plot(plottime,errorvecm,'k'); 
end

%Linear example errors:
if(IC==1)
    fig_err = figure(fig_count);
    fig_count = fig_count +1;
    set(fig_err, 'Name','Error of Run','NumberTitle','off');
    plot(plottime,errorveci,'r'); hold on;
    plot(plottime,errorvecr,'b');
    plot(plottime,errorvecm,'k'); hold off;
    errorsr(hi) = mean(errorvecr);%./length(errorvecr);
    errorsi(hi) = mean(errorveci);%./length(errorveci);
    errorsm(hi) = mean(errorvecm);%./length(errorvecm);
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

if(track_vortices==1 && IC==3)
       
   %Setup velocity vectors:
   VY = zeros(size(v_trackY));
   VX = VY;
   
   %Get time step of plots:
   plot_dt = plottime(2)-plottime(1);
   
   %Calculate velocities using second-order differencing
   for vi=1:num_vort
       VY(vi,2:end-1) = (v_trackY(vi,3:end) - v_trackY(vi,1:end-2))./(2*plot_dt);
       VY(vi,1)       = (v_trackY(vi,2)     - v_trackY(vi,1))./plot_dt;
       VY(vi,end)     = (v_trackY(vi,end)   - v_trackY(vi,end-1))./plot_dt;       
       VX(vi,2:end-1) = (v_trackX(vi,3:end) - v_trackX(vi,1:end-2))./(2*plot_dt);
       VX(vi,1)       = (v_trackX(vi,2)     - v_trackX(vi,1))./plot_dt;
       VX(vi,end)     = (v_trackX(vi,end)   - v_trackX(vi,end-1))./plot_dt;       
   end
   
   %Record radius between vortices for MSD tests:
   if(length(rv)>1)
      msd_rad(ri,bi) = (abs(max(sqrt(v_trackX(1,:).^2 + v_trackY(1,:).^2)) - min(sqrt(v_trackX(1,:).^2 + v_trackY(1,:).^2)))+ ...
                        abs(max(sqrt(v_trackX(2,:).^2 + v_trackY(2,:).^2)) - min(sqrt(v_trackX(2,:).^2 + v_trackY(2,:).^2))))/2;
   end
   
   %Plot X-Y traced positions:
   qs = 1; %skip distance for better plot
   fig_xytrace = figure(fig_count);   fig_count = fig_count+1;
   if(num_vort==2)
    symbol_list = ['sk';'ob'];   
    for vi=1:num_vort     
      plot(v_trackX(vi,1:qs:end),v_trackY(vi,1:qs:end),symbol_list(vi,:),'LineWidth',2);
      if(vi==1)
         hold on;  
      end
      %quiver2(v_trackX(vi,1:qs:end),v_trackY(vi,1:qs:end),VX(vi,1:qs:end),VY(vi,1:qs:end),'n=',5,'s=','screen','c=',[0 0 0],'w=',[2 2]);
    end
   else
    for vi=1:num_vort     
      plot(v_trackX(vi,1:qs:end),v_trackY(vi,1:qs:end),'ok','LineWidth',1);
      if(vi==1)
         hold on;  
      end
      %quiver2(v_trackX(vi,1:qs:end),v_trackY(vi,1:qs:end),VX(vi,1:qs:end),VY(vi,1:qs:end),'n=',5,'s=','screen','c=',[0 0 0],'w=',[2 2]);
    end
   end   
   xlabel('x(t)','FontSize',fsize); ylabel('y(t)','FontSize',fsize);
   set(gca,'Fontsize',fsize);
   axis('equal');
   axis([xmin xmax ymin ymax]);  
   %legend('Vortex #1 X','Vortex #2 X','Vortex #1 Y','Vortex #2 Y');
   hold off;
   
   %Save x-y trace for MSD fig:
   if(length(rv)==1)
      set(fig_xytrace, 'PaperPositionMode', 'auto');       
      print('-djpeg','-r125',['msd_2vort_BC',num2str(BC),'_xvsy',num2str(endt)],['-f',num2str(fig_xytrace)]); 
      print('-depsc','-r125',['msd_2vort_BC',num2str(BC),'_xvsy',num2str(endt)],['-f',num2str(fig_xytrace)]); 
   end  
   
   %Plot positional graph X vs Y:   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   figpos = figure(fig_count);  fig_count = fig_count+1;
   symbol_list = ['o','s','s','o'];
   qs = 4; %skip distance for better plot
   for vi=1:num_vort
       vortices(1,vi)
       if(vortices(1,vi)==-1)
           marker1 = ['ob']; 
       elseif(vortices(1,vi)==1)
           marker1 = ['sk'];  
       end       
       plot(v_trackX(vi,1:qs:end),v_trackY(vi,1:qs:end),marker1,'LineWidth',2);
       if(vi==1); hold on; end                    
   end   
   hold off;
   set(gcf,'Name','Vortex ring position trace');   
   axis('equal');
   axis([xmin xmax ymin ymax]);
   set(gca,'Fontsize',fsize);
   xlabel('x(t)','Fontsize',fsize);
   ylabel('y(t)','Fontsize',fsize);
      
   figure(figpos)
   fnpos = ['ZvsY'];
   set(gcf, 'PaperPositionMode', 'auto');       
   print('-djpeg','-r125',fnpos,['-f',num2str(figpos)]); 
   print('-depsc','-r125',fnpos,['-f',num2str(figpos)]); 
   saveas(figpos,[fnpos,'.fig']);   
   
   %Plot X-position versus time:
   fig_xvst = figure(fig_count);   fig_count = fig_count+1;
   if(num_vort==2)
   symbol_list = ['-k ';'--b'];
   for vi=1:num_vort
      v_trackXi = v_trackX(vi,:);
      xinit     = v_trackXi(1);
      plot(plottime,xinit*ones(size(v_trackXi)),'-r');
      if(vi==1)
         hold on;  
      end
      if(endt>5000)
          lw = 3; 
      else
          lw = 5;
      end
      plot(plottime,v_trackX(vi,:),symbol_list(vi,:),'LineWidth',lw);
     % maxxdeflect = max(abs(v_trackX(vi,:)-xinit));     
   end
   else
    for vi=1:num_vort
      v_trackXi = v_trackX(vi,:);
      xinit     = v_trackXi(1);
      plot(plottime,xinit*ones(size(v_trackXi)),'-r');
      if(vi==1)
         hold on;  
      end
      plot(plottime,v_trackX(vi,:),'-k','LineWidth',5);
     % maxxdeflect = max(abs(v_trackX(vi,:)-xinit));     
    end
   end
   xlabel('t','FontSize',fsize); ylabel('x(t)','FontSize',fsize);
   set(gca,'Fontsize',fsize);
   if(endt>7000)
       axis([7000 max(plottime) xmin xmax]);
   else
       axis([0 max(plottime) xmin xmax]);
   end
   hold off;   
   
   if(length(rv)==1)
      set(fig_xvst, 'PaperPositionMode', 'auto');       
      print('-djpeg','-r125',['msd_2vort_BC',num2str(BC),'_xvst',num2str(endt),'.jpg'],['-f',num2str(fig_xvst)]); 
      print('-depsc','-r125',['msd_2vort_BC',num2str(BC),'_xvst',num2str(endt),'.eps'],['-f',num2str(fig_xvst)]); 
   end
   
   %Plot radius from center (0,0):
   fig_rvst = figure(fig_count);   fig_count = fig_count+1;
   symbol_list = ['-k ';'--b'];
   if(num_vort==2)
     for vi=1:num_vort
      plot(plottime,sqrt(v_trackX(vi,:).^2 + v_trackY(vi,:).^2),symbol_list(vi,:),'LineWidth',5);  
      if(vi==1)
         hold on;  
      end
     end
   else
     for vi=1:num_vort
      plot(plottime,sqrt(v_trackX(vi,:).^2 + v_trackY(vi,:).^2),'-','LineWidth',5);      
      if(vi==1)
         hold on;  
      end
     end
   end
   xlabel('t','FontSize',fsize); ylabel('r','FontSize',fsize);
   set(gca,'Fontsize',fsize);
   axis([0 max(plottime) 0 xmax]);
   hold off;
   
   if(length(rv)==1)
      set(fig_rvst, 'PaperPositionMode', 'auto');       
      print('-djpeg','-r125',['msd_2vort_BC',num2str(BC),'_rvst',num2str(endt),'.jpg'],['-f',num2str(fig_rvst)]); 
      print('-depsc','-r125',['msd_2vort_BC',num2str(BC),'_rvst',num2str(endt),'.eps'],['-f',num2str(fig_rvst)]); 
   end
      
%    %Plot y-velocity
%    figure(fig_count);  fig_count = fig_count+1;
%    halftimei = floor(length(plottime)/2);  
%    hold on;
%    plot(plottime,VY(1,:),'ob','LineWidth',2);   
%    %plot(plottime,c*ones(size(plottime)),'-r','LineWidth',2);
%    hold off;
%    title('Vortex Y-Velocity');
%    axis([0 max(plottime) -1 0]);
%    xlabel('Time')
%    ylabel('c = dY/dt')    
%    %Compute y-velocity:
%    yvelCoMtmp  = ezfit(plottime,v_trackY(1,:),'poly1');
%    yvelCoM     = yvelCoMtmp.m(2);   
%    yvelgridtmp = ezfit(plottime,yv(1,:),'poly1');
%    yvelgrid    = yvelgridtmp.m(2);   
%    disp('Y Velocity');   
%    disp(['CofM: ',num2str(yvelCoM)]);%, '  Percent Error: ',num2str(zvelCoMerr(rmaxi,bi)),'%']);
%    disp(['Grid: ',num2str(yvelgrid)]);%,'  Percent Error: ',num2str(zvelgriderr(rmaxi,bi)),'%']);  

   %Plot y-position vs time:
   figure(fig_count);  fig_count = fig_count+1;
   hold on;
   if(num_vort==2)
     symbol_list = ['sk';'ob'];
     for vi=1:num_vort    
       plot(plottime,v_trackY(vi,:),'-r','LineWidth',2);
       plot(plottime,yv(vi,:),symbol_list(vi,:),'LineWidth',2,'MarkerSize',6);
     end
   else
     for vi=1:num_vort    
       plot(plottime,v_trackY(vi,:),'-r','LineWidth',2);
       plot(plottime,yv(vi,:),'ok','LineWidth',2,'MarkerSize',6);
     end   
   end    
   hold off;
   title('Vortex Y-Position vs Time');
   axis([0 max(plottime) ymin ymax]);   
   xlabel('t')
   ylabel('y(t)')
  
end %trackvortices

end %hv stepsizes

%Record msd errors:
if(length(rv)>1)
    msd_errors(ri,bi) = max(errorsm);
end
ri = ri+1;
end %rv  

if(length(rv)>1) 
 cv = ['-b ';'--r';'og ';'-k ';'-c ']; 
 fig_msd   = figure(fig_count);
 fig_count = fig_count+1;
 if(BC==2)
     semilogy(rv_msd,msd_errors(1:length(rv_msd),bi),cv(bi,:),'LineWidth',4,'MarkerSize',4);
 else
     semilogy(rv,msd_errors(1:length(rv),bi),cv(bi,:),'LineWidth',4,'MarkerSize',4);
 end
 xlim([min(rv) max(rv)]);
 xlabel('r','Fontsize',fsize);
 ylabel('Average Max Error of Re(\Psi) and Im(\Psi)','Fontsize',fsize);
 set(gca,'Fontsize',fsize);
 hold on;
 
 if(track_vortices==1)
  cv = ['-b ';'--r';'-c ']; 
 fig_msd2  = figure(fig_count);
 fig_count = fig_count+1;
 if(BC==2)
    plot(rv_msd,100*msd_rad(1:length(rv_msd),bi)./d,cv(bi,:),'LineWidth',3);
 else
    plot(rv,100*msd_rad(1:length(rv),bi)./d,cv(bi,:),'LineWidth',3);
 end
 xlim([12 max(rv)]);
 xlabel('d','Fontsize',fsize);
 ylabel('Variation of radius (%)','Fontsize',fsize);
 set(gca,'Fontsize',fsize);
  hold on;  
 end
end

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
    if(method == 1)
        strtmpm1 = ['CD:     m = ',num2str(mav,3)];
        strtmp2 = '--sb';
    elseif(method == 2)
        strtmpm2 = ['2SHOC:  m = ',num2str(mav,3)];
        strtmp2 = '-ok';
    elseif(method == 13)
        strtmp = ['RHOC:  m = ',num2str(mav,3)];
    elseif(method == 15)
        strtmp = ['RHOCNEW:  m = ',num2str(mav,3)];
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
    xlabel('h','Fontsize',18);
    set(gca,'XLim',[min(hv)-min(hv)/2, max(hv)+max(hv)/2]);
    set(gca,'XTick',fliplr(hv));
    set(gca,'XTickLabel',{'1/16','1/8','1/3','1/2','1'});
    ylabel('Ave Error Norm','Fontsize',18);
    end
end %IC=1

numframetimes(ni)     = tcomp;
numframechunksize(ni) = chunk_size;

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
        if(xreswanted == floor(sqrt(1000)))
            mstr = 'r--o';
        elseif(xreswanted == floor(sqrt(10000)))
            mstr = 'r--s';
        elseif(xreswanted == floor(sqrt(100000)))
            mstr = 'r--^';
        elseif(xreswanted == floor(sqrt(1000000)))
            mstr = 'r--d';
        end               
    else
        if(xreswanted == floor(sqrt(1000)))
            mstr = 'k-o';
        elseif(xreswanted == floor(sqrt(10000)))
            mstr = 'k-s';
        elseif(xreswanted == floor(sqrt(100000)))
            mstr = 'k-^';
        elseif(xreswanted == floor(sqrt(1000000)))
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
end %numframes    

end %BC

if(length(rv)>1 && length(BCv)>1)
   set(fig_msd, 'PaperPositionMode', 'auto');  
   print('-djpeg','-r125',['MSD_2D_T',num2str(endt),'_k',num2str(k),'_h',num2str(h),'_r',num2str(min(rv)),'-',num2str(max(rv)),'_f',num2str(numframes),'.jpg'],['-f',num2str(fig_msd)]); 
   print('-depsc','-r125',['MSD_2D_T',num2str(endt),'_k',num2str(k),'_h',num2str(h),'_r',num2str(min(rv)),'-',num2str(max(rv)),'_f',num2str(numframes),'.eps'],['-f',num2str(fig_msd)]); 
   if(track_vortices==1)
      set(fig_msd2, 'PaperPositionMode', 'auto');  
      print('-djpeg','-r125',['MSD_2D_T',num2str(endt),'_k',num2str(k),'_h',num2str(h),'_r',num2str(min(rv)),'-',num2str(max(rv)),'_f',num2str(numframes),'.jpg'],['-f',num2str(fig_msd2)]); 
      print('-depsc','-r125',['MSD_2D_T',num2str(endt),'_k',num2str(k),'_h',num2str(h),'_r',num2str(min(rv)),'-',num2str(max(rv)),'_f',num2str(numframes),'.eps'],['-f',num2str(fig_msd2)]); 
   end
end



%Save CUDA timing result in file:
if(((length(cudav)>1)||length(methodv)==4)  && length(xreswantedv)>1)    
       fprintf(fid, '%7.3f\n', tcomp);  
end
end %xreswanted

if(((length(cudav)>1)||length(methodv)==4)  && length(xreswantedv)>1) 
   for ggg=1:length(xreswantedv)       
       fprintf(fid, '%7d\n',   xreswantedv(ggg));        
   end
   fclose(fid);
end

end %method
if(length(methodv)>1 && length(hv)>1)
    legend(strtmpm1,strtmpm2)
    hold off;
end

end %IC
end %precision

if(length(numframesv)>1)
   figure(fig_frames);
   hold off;
   set(gcf, 'PaperPositionMode', 'auto');       
   print('-djpeg','-r125',['2D_T',num2str(endt),'_N',num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),'_M',num2str(method),'.jpg'],['-f',num2str(fig_frames)]); 
   print('-depsc','-r125',['2D_T',num2str(endt),'_N',num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),'_M',num2str(method),'.eps'],['-f',num2str(fig_frames)]); 
end
end %cuda


%Report wall time:
twall = toc(twall);
disp(['NLSE2D Total Wall Time: ',num2str(twall),' seconds.']);

%Close diary file for CUDA runs:
if(((length(cudav)>1)||length(methodv)==4) && length(xreswantedv)>1)
    diary off;
end

%Exit MATLAB is desired (good for nohup/profilers
if(exit_on_end==1),  exit; end;

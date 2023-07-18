%NLSEmagic1D_FRS  Full Research Driver Script for NLSEmagic1D
%Program to integrate the one-dimensional Nonlinear Shrodinger Equation:
%i*Ut + a*Uxx - V(x)*U + s*|U|^2*U = 0.
%
%2014 Ronald M Caplan
%Developed with support from the
%Computational Science Research Center
%San Diego State University.
%
%Distributed as part of the NLSEmagic software package at:
%http://www.nlsemagic.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  VERSION:  2.0    2014     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Clear any previous run variables and close all figures:
munlock; close all; clear all; clear classes; clear memory;
twall = tic; %Used to calculate total wall time.

%------------------Simulation parameters----------------------
endtw        = 5;     %Desired end time fo the simulation - may be slightly altered.
chunksizev   = [0];    %Run CUDA codes with different chunksizes.
numframesv   = [1000];  %Number of times to view/plot/analyze solution.
hv           = [1/50]; %Spatial step-size.
k            = 0;      %Time step-size.  [Set to 0 to auto-compute smallest stable timestep]
ICv          = [1];    %Select intial condition (1: bright soliton, 2: dark, 3: linear gaussian, 4: gray soliton)
methodv      = [11];    %Spatial finite-difference scheme: 1:CD O(h^2), 2:2SHOC O(h^4), 11:Script CD, 12:Script 2SHOC.
runs         = 1;      %Number of times to run simulation.
xreswantedv  = [500];    %Desired grid size (N) (Overwrites IC auto grid-size calculations, and may be slightly altered)
cudav        = [0];    %Use NVIDIA GPU-acceleration CUDA codes (1) or CPU (0)
precisionv   = [2];    %Use single (1) or double (2) precision integrators.
BCv          = [0];    %Overides boundary condition choice of IC (0:use IC BC 1:Dirichlet 2:MSD 3:Lap0 4:1-side 5:exact (IC2 only)
rv           = [0];    %Used in MSD tests (Distance from soliton center to grid boundary).
tol          = 10;     %Modulus-squared tolerance to detect blowup.
%------------------Analysis switches----------------------
calc_mass    = 1;      %Calculates change in mass.
add_pert     = 0.0;    %Add uniform random perturbation of size e=add_pert.
%------------------Simulation switches--------------------
pause_init   = 0;      %Pause initial condition before integation.
exit_on_end  = 0;      %Close MATLAB after simulation ends.
%------------------Plotting switches----------------------
show_waitbar = 0;      %Show waitbar with completion time estimate.
plotsol      = 1;      %Plot solution at every frame.
%------------------Output switches----------------------
save_movies  = 0;      %Generate a gif (1) or avi (2) movie of simulation.
save_images  = 0;      %Gernate eps images of simulation.
disp_pt_data = 0;      %Output value of Psi(5) at each frame.
%---------------------------------------------------------

%Set filenames for CUDA timing run based on installed GPU:
if(((length(cudav)>1)||length(methodv)>1) && length(xreswantedv)>1)
  str = computer;
  if(strcmp(str, 'PCWIN'))
      strcuda = 'GT430';
  elseif(strcmp(str, 'PCWIN64'))
      strcuda = 'GTX580';
  elseif(strcmp(str, 'GLNX86'))
      strcuda = 'GTX670';
  else
      strcuda = 'UNKNOWN';
  end
  cudafn1 =  [strcuda,'_1D_RES', num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),...
              '_T',num2str(ceil(endtw/k)),'_Fr',num2str(max(numframesv))];
  diary([cudafn1,'.txt']);
end

%Start parameter loops - for single run, all vectors are of length 1.
for cuda=cudav
for precision = precisionv
for IC=ICv
for method=methodv
if(((length(cudav)>1)||length(methodv)>1) && length(xreswantedv)>1)
  cudafn =  [cudafn1,'_IC',num2str(IC),'_M',num2str(method),'_P',...
                 num2str(precision),'_CUDA',num2str(cuda),'.txt'];
  fid = fopen(cudafn,'wt');
end
for xreswanted=xreswantedv

bi = 0;
for BC_i=BCv
bi = bi+1;
ni = 0;
numframetimes     = zeros(size(numframesv)); %CUDA chunksize tests
numframechunksize = zeros(size(numframesv)); %CUDA chunksize tests
for numframes = numframesv
ni = ni+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION PARAMETERS        %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
if(IC==1)%Bright sech soliton
   a    = 1;
   s    = 1;
   OM   = 0.5;   %Frequency
   c    = 0.75;  %Velocity of soliton
   x0   = 0;
   %Compute numerical domain to be big enough to avoid BC effects:
   xmin = -sqrt(a/OM)*asech(eps*sqrt(s/(2*OM)));
   xmax =  sqrt(a/OM)*asech(eps*sqrt(s/(2*OM))) + c*endtw;
   BC   = 1;   %Dirchilet BC
elseif(IC==2)%Tanh dark soliton
   a    =  1;
   s    = -1;
   OM   = -1;   %Frequency
   c    = 0.75; %Velocity of soliton
   x0   = 0;
   %Compute numerical domain to be big enough to avoid BC effects:
   xmin = -sqrt(abs((2*a)/OM))*atanh(1-(eps)*sqrt(s/OM));
   xmax =  sqrt(abs((2*a)/OM))*atanh(1-(eps)*sqrt(s/OM)) + c*endtw;
   BC   = 2;   %Modulus-Squared Dirchilet BC
elseif(IC==3)%Linear exponential
   a    = 1;
   s    = 0;
   xmin = -sqrt(-log(sqrt(eps))*2*a);
   xmax =  sqrt(-log(sqrt(eps))*2*a);
   BC   = 1;   %Dirchilet BC
elseif(IC==4)%Grey soliton
   a    =  1;
   s    = -1;
   OM   = -1;   %Frequency
   c    = 0.75; %Velocity of soliton
   x0   = 0;
   %Compute numerical domain to be big enough to avoid BC effects:
   xmin = -sqrt(abs((2*a)/OM))*atanh(1-sqrt(eps)*sqrt(s/OM));
   xmax =  sqrt(abs((2*a)/OM))*atanh(1-sqrt(eps)*sqrt(s/OM)) + c*endtw;
   BC   = 2;   %Modulus-Squared Dirchilet BC
end

%Manual overide of domain sizes:
%xmax = 15;  xmin = -xmax;


%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp('----------------------------------------------------');
disp('--------------------NLSE1D--------------------------');
disp('----------------------------------------------------');
disp('Checking parameters...');

%Override BC from IC in case of specified BCs:
if(BCv(1)~=0)
   BC = BC_i;
end
%Start MSD test loop:
ri = 1;
for rmax=rv
if(rmax~=0 || length(rv)>1)
 xmax = rmax + c*endtw;
 xmin = -rmax;
 disp(['ri: ',num2str(ri)]);
end

%Step size loop (for scheme order analysis)
hi=0;
for h = hv
hi = hi+1;

%Adjust ranges so they are in units of h (fixes bugs):
xmin = xmin - rem(xmin,h);
xmax = xmax - rem(xmax,h);
%Update rv accordingly
if(rmax~=0 || length(rv)>1)
   rv(ri) = rv(ri)-max(abs([rem(xmin,h), rem(xmax,h)]));
end

%Overide limits for CUDA timings:
if(xreswanted>0)
    xmax =  (xreswanted-1)*h/2;
    xmin = -(xreswanted-1)*h/2;
end

%Set up spatial grid:
xvec = (xmin:h:xmax)';
xres = length(xvec);

%Set up CUDA info
if(cuda==1)
   cudablocksize     = 512;
   sharedmemperblock = (cudablocksize+2)*(3+2*method)*(4*precision)/1000;
   numcudablocks     = ceil(xres/cudablocksize);
   %For MSD BC, need to check this:
   if(BC==2)
      if(xres - cudablocksize*(numcudablocks-1) == 1)
         disp('MSD CUDA ERROR: N (x) is one cell greater than CUDA block,')
         xmax = xmax-h;
         xmin = xmin+h;
         disp(['adjusting xmax to ',num2str(xmax), ' and xmin to ',num2str(xmin),' to compensate.']);
       end
       %Now, recompute new grid:
       xvec  = (xmin:h:xmax)';  xres = length(xvec);
       numcudablocks  = ceil(xres/cudablocksize);
   end
end

for run_i=1:runs %Run loop (for averaging timings)

%Initialize solution and potential matrices:
U = zeros(xres,1);
V = zeros(size(xvec));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION FORMULATION       %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
if(IC==1)
   V      = 0.*ones(size(xvec));
   U      = sqrt((2*OM)/s).*sech(sqrt(OM/a).*(xvec - x0)).*exp(1i*((c/(2*a))*xvec));
elseif(IC==2)
   V      = 0.*ones(size(xvec));
   U      = sqrt(OM/s).*tanh(sqrt(abs(OM/(2*a))).*(xvec-x0)).*exp(1i*((c/(2*a))*xvec));
elseif(IC==3)
   V   = xvec.^2/a;
   U   = exp(-xvec.^2/(2*a));
elseif(IC==4)
   V      = 0.*ones(size(xvec));
   vc     = sqrt(-OM*2*a);
   E      = sqrt(a/-OM);
   U      = sqrt(OM/s)*(1i*(c/vc) + sqrt(1-c^2/vc^2)*tanh((xvec./(sqrt(2)*E))*sqrt(1-c^2/vc^2)));
end

%Add random perturbation if specified:
if(add_pert>0)
    U = U.*(1 + add_pert.*rand(size(U)));
end

%-------------------------------------------------------------------------%
%Compute stability bounds for time step k:
if(k==0)
   hmin = min(hv);
   if(method==1 || method==11)
      G = [4,3,1,0];
      klin = hmin^2/(sqrt(2)*a);
   elseif((method==2 || method==12) || length(methodv)>1)
      G = (1/12)*[64,63,46,12,-3,-4];
      klin = (3/4)*hmin^2/(sqrt(2)*a);
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
       UB_1  = [U(2);U(end-1)];
       Ut    = NLSEmagic1D_FRS_F_MAT(U,V,hmin,s,a,method,BC,(1/hmin^2),(a/12),(a/(6*hmin^2)),(1/a));
       UtB_1 = [Ut(2);Ut(end-1)];
       Bmax = real(max((hmin^2/(1i*a))*UtB_1./UB_1));
       clear UtB_1 UB_1 Ut;
    elseif(BC==3)
       UB = [U(1);U(end)];
       VB = [V(1);V(end)];
       Bmax = max((hmin^2/a)*(s*UB.*conj(UB) - VB));
       clear VB UB;
   else
       Bmax = 0;
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
steps = floor(endt/k); %<--Compute number of steps required
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
disp('1D NLSE Parameters:')
disp(['a:          ',num2str(a)]);
disp(['s:          ',num2str(s)]);
disp(['xmin:       ',num2str(xmin)]);
disp(['xmax:       ',num2str(xmax)]);
disp(['Endtime:    ',num2str(endt)]);
disp('Numerical Parameters:');
disp(['h:          ',num2str(h)]);
disp(['k:          ',num2str(k)]);
disp(['GridSize:   ',num2str(xres)]);
disp(['TimeSteps:  ',num2str(steps)]);
disp(['ChunkSize:  ',num2str(chunk_size)]);
disp(['NumFrames:  ',num2str(numframes)]);
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
    disp('Boundary Conditions:  MSD');
elseif(BC==3)
    disp('Boundary Conditions:  Uxx = 0');
elseif(BC==4)
    disp('Boundary Conditions:  One-Sided Diff');
elseif(BC==5)
    disp('Boundary Conditions:  Exact');
end
if(cuda == 1)
    disp( 'CUDA Parameters and Info:');
    disp(['BlockSize:     ',num2str(cudablocksize)]);
    disp(['CUDAGridSize:  ',num2str(numcudablocks)]);
    disp(['NumBlocks:     ',num2str(numcudablocks)]);
    disp(['Shared Memory/Block: ',num2str(sharedmemperblock),'KB']);
    disp(['TotalGPUMemReq:      ',num2str(xres*(9 + 2*(method-1))*(4*precision)/1024),'KB']);
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
if(length(hv)>1)
    disp(['h:          ',num2str(h)]);
    disp(['k:          ',num2str(k)]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             PLOTTING SETUP                      %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
fsize       = 20;  %Set font size for figures.
fig_count   =  1;  %Used to create figure numbers.
if(plotsol==1)
%Find maximums for plot axis:
maxMod2U  = max(U.*conj(U));
maxRealU  = max(real(U));
plotmax   = max([maxMod2U maxRealU]);

%Set up plot of initial condition:
fig_plot  = figure(fig_count);
fig_count = fig_count+1;
set(fig_plot, 'Name','1D NLSE  t = 0','NumberTitle','off');
plot_real = plot(xvec,real(U),   '-b' ,'LineWidth',4);
hold on;
%plot_v = plot(xvec,V,   '-g');
plot_imag = plot(xvec,imag(U),   '--r' ,'LineWidth',4);
plot_mod2 = plot(xvec,U.*conj(U),'-k'  ,'LineWidth',6);
%plot_true = plot(xvec,U.*conj(U),'-og');
%plot_phase = plot(xvec,angle(U),'--c');
hold off;
axis([xmin xmax (-plotmax-0.1*plotmax) (plotmax + 0.1*plotmax)]);
xlabel('x','Fontsize',fsize);
ylabel('\Psi(x)','Fontsize',fsize);
set(gca,'Fontsize',fsize);
%legend('|\Psi_e_x_a_c_t|^2','Re(\Psi)','Im(\Psi)','|\Psi|^2','Phase','Location','SouthWest');
drawnow;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(save_movies>=1 || save_images==1)
   fn = ['IC',num2str(IC),'_'];
end

%Setup waitbar:
if(show_waitbar==1)
    fig_waitbar = waitbar(0,'Estimated time to completion ???:??');
    set(fig_waitbar, 'Name','Simulation 0% Complete','NumberTitle','off');
    waitbar(0,fig_waitbar,'Estimated time to completion ???:??');
end

%Pause simulation if desired.
if(pause_init==1)
    disp('Initial condition displayed.  Press a key to start simulation');
    pause
end

%Setup movie files:
if(save_movies==1)
    gifname = [fn,'.gif'];
    I = getframe(fig_plot);
    I = frame2im(I);
    [XX, map] = rgb2ind(I, 128);
    imwrite(XX, map, gifname, 'GIF', 'WriteMode', ...
       'overwrite', 'DelayTime', 0, 'LoopCount', Inf);
elseif(save_movies==2)
   mov_disk = avifile([fn,'.avi'],'compression','None','quality',100);
end

%Save image of IC:
if(save_images==1)
   set(fig_plot,'InvertHardcopy','on')
   set(fig_plot, 'PaperPositionMode', 'auto');  %Makes better images
   fignum = ['-f',num2str(fig_plot)];
   print('-depsc','-r100',[fn '_t0.eps'],fignum);
end


%Initialize counters and timers:
calcn  = 0;
n      = 0;
ttotal = tic; %Start total time timer
tcomp  = 0;   %Initialize compute-time to 0

%Initialize RK4 matrices for script integrators:
if(method>10)
   Uk_tmp = zeros(size(U));
   k_tot  = zeros(size(U));
end

%Initialize error plot vectors:
plottime  = zeros(numframes,1);
errorveci = zeros(numframes,1);
errorvecr = zeros(numframes,1);
errorvecm = zeros(numframes,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             BEGIN SIMULATION                    %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
for chunk_count = 1:numframes

%Start chunk-time timer (includes plots)
if(show_waitbar==1), tchunk  = tic; end
tcompchunk = tic;  %Start compute-time timer

%Reset methods for BC 5 since not implemented in mex:
if((BC==5) && (method==1 || method==2))
    method=method+10;
    disp('Warning, using script integraters due to exact BC choice!');
end

%Call MEX integrator codes:
if(method<10)
  if(cuda==0)
    U = NLSEmagic1D_Take_Steps_CPU(U,V,s,a,h^2,BC,chunk_size,k,precision,method);
  else
    U = NLSEmagic1D_Take_Steps_GPU(U,V,s,a,h^2,BC,chunk_size,k,precision,method);
  end
else  %Old code for trial methods / mex timings
    %Do divisions first to save compute-time:
    k2   = k/2;    k6   = k/6;    l_a  = 1/a;    l_h2 = 1/h^2;
    l76  = 7/6;    l_12 = 1/12;

    for nc = 1:chunk_size
    %Need these params for BC=5:
    if(BC==5)
        par = [k*(n+nc-1);xvec(1);xvec(end);OM;c];
    else
        par = [];
    end
    %Start Runga-Kutta:
    k_tot  = NLSEmagic1D_FRS_F_MAT(U,V,h,s,a,method,BC,l_a,l_h2,l76,l_12,par); %K1
    %----------------------------
    Uk_tmp = U + k2*k_tot;
    par(1) = k*(n+nc-1) + k2;
    Uk_tmp  = NLSEmagic1D_FRS_F_MAT(Uk_tmp,V,h,s,a,method,BC,l_a,l_h2,l76,l_12,par); %K2
    k_tot = k_tot + 2*Uk_tmp; %K1 + 2K2
    %----------------------------
    Uk_tmp  = U + k2*Uk_tmp;
    Uk_tmp  = NLSEmagic1D_FRS_F_MAT(Uk_tmp,V,h,s,a,method,BC,l_a,l_h2,l76,l_12,par); %K3
    k_tot = k_tot + 2*Uk_tmp; %K1 + 2K2 + 2K3
    %----------------------------
    Uk_tmp  = U + k*Uk_tmp;
    par(1)  = k*(n+nc) + k;
    k_tot   = k_tot + NLSEmagic1D_FRS_F_MAT(Uk_tmp,V,h,s,a,method,BC,l_a,l_h2,l76,l_12,par); %K1 + 2K2 + 2K3 + K4
    %-------------------------------
    U = U + k6*k_tot;   %New time step:

    if(BC==5)%Exact BC solution for IC 3
        U(1)   = sqrt(OM/s).*tanh(sqrt(abs(OM/(2*a))).*(xvec(1)   - c*k*(n+nc))).*exp(1i*((c/(2*a))*xvec(1)   + (OM - c^2/(4*a))*k*(n+nc)));
      U(end)   = sqrt(OM/s).*tanh(sqrt(abs(OM/(2*a))).*(xvec(end) - c*k*(n+nc))).*exp(1i*((c/(2*a))*xvec(end) + (OM - c^2/(4*a))*k*(n+nc)));
    end
    end %chunksize
end %method

%Add to compute-time counter and update step number:
tcomp = tcomp + toc(tcompchunk);
n = n + chunk_size;

if(plotsol==1)
%Set fig titles first, so know when possible crash happens:
   set(fig_plot, 'Name',['1D NLSE  t = ',num2str((n*k),'%.2f')]);
end

%Detect blow-up:
if(max(real(U(:))) > tol);% || sum(isnan(U(:))) > 0)
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

%Compute true solution to compare error:
if(IC==1)
    Ureal = sqrt((2*OM)/s).*sech(sqrt(         OM/a).*(xvec - x0 - c*n*k)).*exp(1i*((c/(2*a))*xvec + (OM - c^2/(4*a))*k*n));
elseif(IC==2)
    Ureal = sqrt(    OM/s).*tanh(sqrt(abs(OM/(2*a))).*(xvec - x0 - c*n*k)).*exp(1i*((c/(2*a))*xvec + (OM - c^2/(4*a))*k*n));
elseif(IC==3)
    Ureal = exp(-xvec.^2/(2*a)).*exp(-1i*k*n);
elseif(IC==4)
    Ureal = sqrt(OM/s)*(1i*(c/(vc)) + sqrt(1-(c^2/(vc)^2))*tanh(((xvec - x0 - c*n*k)./(sqrt(2)*E))*sqrt(1-(c^2/(vc)^2)))).*exp(1i*OM*k*n);
end

if(plotsol==1)
%Plot current time step:
set(plot_mod2,'ydata',U.*conj(U));
set(plot_real,'ydata',real(U));
set(plot_imag,'ydata',imag(U));
%set(plot_true,'ydata',real(Ureal));
%set(plot_phase,'ydata',angle(U));
drawnow;
end

%Now save movie frame or image:
if(save_movies==1)
    gifname = [fn,'.gif'];
    I = getframe(fig_plot);
    I = frame2im(I);
    [XX, map] = rgb2ind(I, 128);
    imwrite(XX, map, gifname, 'GIF', 'WriteMode',...
                'append', 'DelayTime', 0);
elseif(save_movies==2)
   mov_disk = addframe(mov_disk,getframe(fig_plot));
end
if(save_images==1)
   set(fig_plot,'InvertHardcopy','on')
   fignum = ['-f',num2str(fig_plot)];
   set(fig_plot, 'PaperPositionMode', 'auto');
   print('-depsc','-r100',[fn '_t' num2str(n*k) '.eps'],fignum);
end

%%%Analysis Tools%%%%%%%%%*********************************************
calcn = calcn+1;
plottime(calcn) = n*k;

%Display solution at grid pioint for CUDA validation:
if(disp_pt_data==1)
    disp(['Re(U(5)): ',num2str(real(U(5)))]);
end

%Calculate mass:
if(calc_mass==1)
   mass(calcn) = sum(U.*conj(U))*h;
   delta_mass = (abs(mass(1)-mass(calcn))/mass(1))*100;
end

%Record error
errorvecr(calcn) = sqrt(sum((real(Ureal(:)) - real(U(:))).^2)       /length(U(:)));
errorveci(calcn) = sqrt(sum((imag(Ureal(:)) - imag(U(:))).^2)       /length(U(:)));
errorvecm(calcn) = sqrt(sum((abs(Ureal(:)).^2  -  abs(U(:)).^2).^2) /length(U(:)));

errorvecr(calcn) = max(abs(real(Ureal(:)) - real(U(:))));
errorveci(calcn) = max(abs(imag(Ureal(:)) - imag(U(:))));
errorvecm(calcn) = max(abs(abs(Ureal(:)).^2  -  abs(U(:)).^2));


% errorvecr(calcn) = sqrt(sum((real(Ureal(1)) - real(U(1))).^2)/length(U(1)));
% errorveci(calcn) = sqrt(sum((imag(Ureal(1)) - imag(U(1))).^2)/length(U(1)));
% errorvecm(calcn) = sqrt(sum((abs(Ureal(1)).^2  -  abs(U(1)).^2).^2)/length(U(1)));

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
   mov_disk = close(mov_disk);
end

end %Repeated runs complete

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
   plot(plottime,mass,'-');
   title_txt = ['\DeltaMass = ',num2str(delta_mass,'%.4f')];
   title(title_txt);
   grid on;
   xlabel('Time');   ylabel('Mass');
end

%Errors
fig_err  = figure(fig_count); fig_count = fig_count+1;
figure(fig_err);
set(fig_err, 'Name','Error of Run','NumberTitle','off');
plot(plottime,errorveci,'r--','LineWidth',3);
hold on;
plot(plottime,errorvecr,'b-','LineWidth',3);
plot(plottime,errorvecm,'k-','LineWidth',5);
xlabel('Time','Fontsize',fsize);
grid on;
ylabel('Ave(|Error|_2)','Fontsize',fsize);
set(gca,'Fontsize',fsize);
errormax = max([errorveci;errorvecr;errorvecm]);
set(gca,'XLim',[0,  plottime(end)]);
set(gca,'YLim',[0, (errormax+0.1*errormax)]);
hold off;
drawnow;
errorsr(hi) = max(errorvecr);%./length(errorvecr);
errorsi(hi) = max(errorveci);%./length(errorveci);
errorsm(hi) = max(errorvecm);%./length(errorvecm);
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
end% hv stepsizes

%Save rerrors for MSD tests:
msd_errors(ri,bi) = max((errorsi(:) + errorsr(:))/2);
ri = ri+1;
end %rv

%Output error results in latex table format:
h_str       = num2str(hv(1));
errorsr_str = num2str(errorsr(1));
errorsi_str = num2str(errorsi(1));
errorsm_str = num2str(errorsm(1));
mr_str = '--';
mi_str = '--';
mm_str = '--';
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
        %mav = mean(mm);
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
    end

    if(exist('fig_order','var')==0)
       fig_order = figure(fig_count); fig_count = fig_count+1;
    else
       figure(fig_order)
    end
    set(fig_order, 'Name','Method Orders','NumberTitle','off');
    loglog(hv,0.5*(errorsr+errorsi),strtmp2,'LineWidth',2)
    hold on;
    %loglog(hs,errorsi,strtmp2i,'LineWidth',2)
    %loglog(hs,errorsm,strtmp2m,'LineWidth',2)
    %legend('Real','Imag');
    %a1 = annotation('textbox',[0.25 0.6 0.5 0.1]);
    %set(a1,'String',strtmp,'Fontsize',18,'FitBoxToText','on');
    set(gca,'Fontsize',fsize);
    xlabel('h','Fontsize',fsize);
    set(gca,'XLim',[min(hv)-min(hv)/2, max(hv)+max(hv)/2]);
    set(gca,'XTick',fliplr(hv));
    set(gca,'XTickLabel',{'1/32','1/16','1/8','1/3','1/2'});
    ylabel('Ave Error Norm','Fontsize',fsize);
end

%Save timings for chunksize tests:
numframetimes(ni)     = tcomp;
numframechunksize(ni) = chunk_size;
end %numframesv

%Make MSD test figure:
if(length(rv)>1)
 cv = ['--b';'sr ';'og ';'-k ';'-c '];
 fig_msd   = figure(fig_count);
 fig_count = fig_count+1;
 semilogy(rv,msd_errors(:,bi),cv(bi,:),'LineWidth',4,'MarkerSize',4);
 xlim([min(rv) max(rv)]);
 xlabel('r','Fontsize',fsize);
 ylabel('Average Max Error of Re(\Psi) and Im(\Psi)','Fontsize',fsize);
 set(gca,'Fontsize',fsize);
 hold on;
end

%Show chunk-size results:
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
        if(xreswanted == 1000)
            mstr = 'r--o';
        elseif(xreswanted == 10000)
            mstr = 'r--s';
        elseif(xreswanted == 100000)
            mstr = 'r--^';
        elseif(xreswanted == 1000000)
            mstr = 'r--d';
        end
    else
        if(xreswanted == 1000)
            mstr = 'k-o';
        elseif(xreswanted == 10000)
            mstr = 'k-s';
        elseif(xreswanted == 100000)
            mstr = 'k-^';
        elseif(xreswanted == 1000000)
            mstr = 'k-d';
        end
    end
    plot(numframechunksize,numframetimes./min(numframetimes),mstr,'MarkerFaceColor','w','LineWidth',3);
    hold on;
    set(gca, 'XScale','log');
    set(gca,'Fontsize',fsize);
    axis([10 400 1 3]);
    %set(gca,'XTick',[10:10:200]);
    set(gca,'XTick',[10 25 50 100 200 400],'XTickLabel',[10 25 50 100 200 400]);
    xlabel('Chunksize','Fontsize',fsize);
    ylabel('Slowdown Factor','Fontsize',fsize);
end

end %BC

%Save figure results for MSD tests:
if(length(rv)>1 && length(BCv)>1)
   figure(fig_msd)
   set(gcf, 'PaperPositionMode', 'auto');
   print('-djpeg','-r100',strrep(['MSD_1D_c',num2str(c),'_T',num2str(endt),'_k',num2str(k),'_h',num2str(h),'_r',num2str(min(rv)),'-',num2str(max(rv)),'_f',num2str(numframes)],'.',''),['-f',num2str(fig_msd)]);
   print('-depsc','-r100',strrep(['MSD_1D_c',num2str(c),'_T',num2str(endt),'_k',num2str(k),'_h',num2str(h),'_r',num2str(min(rv)),'-',num2str(max(rv)),'_f',num2str(numframes)],'.',''),['-f',num2str(fig_msd)]);
end

%Save CUDA timing result in file:
if(((length(cudav)>1)||length(methodv)>1)  && length(xreswantedv)>1)
       fprintf(fid, '%7.3f\n', tcomp);
end

end %xreswanted

%Save xres vector values in file for CUDA timing results:
if(((length(cudav)>1)||length(methodv)>1) && length(xreswantedv)>1)
   for ggg=1:length(xreswantedv)
       fprintf(fid, '%7d\n',   xreswantedv(ggg));
   end
   fclose(fid);
end

end %method

%Add legend to order-of-accuracy results:
if(length(methodv)>1 && length(hv)>1)
    figure(fig_order);
    legend(strtmpm1,strtmpm2)
    hold off;
end

end %IC
end %precision

%Save figures of chunksize results:
if(length(numframesv)>1)
   figure(fig_frames);
   hold off;
   set(gcf, 'PaperPositionMode', 'auto');
   print('-djpeg','-r100',strrep(['1D_T',num2str(endt),'_N',num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),'_M',num2str(method)],'.',''),['-f',num2str(fig_frames)]);
   print('-depsc','-r100',strrep(['1D_T',num2str(endt),'_N',num2str(min(xreswantedv)),'_',num2str(max(xreswantedv)),'_M',num2str(method)],'.',''),['-f',num2str(fig_frames)]);
end

end %cuda

%Report wall time:
twall = toc(twall);
disp(['NLSE1D Total Wall Time: ',num2str(twall),' seconds.']);

%Close diary file for CUDA runs:
if(((length(cudav)>1)||length(methodv)>1) && length(xreswantedv)>1)
    diary off;
end

%Exit MATLAB is desired (good for nohup/profilers)
if(exit_on_end==1),  exit; end;

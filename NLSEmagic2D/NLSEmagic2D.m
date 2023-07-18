%NLSEmagic2D
%Program to integrate the two-dimensional Nonlinear Shrödinger Equation:
%i*Ut + a*(Uxx+Uyy) - V(x,y)*U + s*|U|^2*U = 0.
%
%©2011 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center at 
%San Diego State University.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  VERSION:  012  1/8/2012   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Clear any previous run variables and close all figures:
clear mex; close all; clear all;
%------------------Simulation paramaters----------------------
endtw       = 5;     %End time of simulation.
numframes   = 50;    %Number of desired frames to plot.
h           = 1/10;  %Spatial grid spacing 
k           = 0;     %Time step [Set to 0 to auto-compute smallest stable timestep]
method      = 1;     %Method:  1: CD O(4,2), 2: 2SHOC O(4,4)
cuda        = 0;     %Try to use CUDA code (if installed and compiled)
precision   = 2;     %Single (1) or double (2) precision.
tol         = 10;    %Simulation mod-squared tolerance to detect blowup

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION PARAMETERS        %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
s  = 0; %Linear
a  = 1; 
BC = 1; %Dirchilet BC
%Compute numerical domain to be big enough to avoid BC effects:
ymax = 0.5*sqrt(-log(sqrt(eps))*2*a);
xmin = -2*ymax;
xmax =  2*ymax;
ymin = -ymax;  
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('----------------------------------------------------');
disp('--------------------NLSEmagic2D---------------------');
disp('----------------------------------------------------');

%Adjust ranges so they are in units of h:
xmin = xmin - rem(xmin,h); ymin = ymin - rem(ymin,h);
xmax = xmax - rem(xmax,h); ymax = ymax - rem(ymax,h);

%Set up grid:
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
            if(M - cudablocksizex*(numcudablocksx-1) == 1)
                disp('MSD CUDA ERROR: M (yres) is one cell greater than CUDA block,')                
                ymax = ymax-h;    
                ymin = ymin+h;
                disp(['adjusting ymax to ',num2str(ymax),' and ymin to ',num2str(ymin),' to compensate,']);                         
            end
            if(N - cudablocksizey*(numcudablocksy-1) == 1) 
                disp('MSD CUDA ERROR: N (xres) is one cell greater than CUDA block')
                xmax = xmax-h;   
                xmin = xmin+h; 
                disp(['adjusting xmax to ',num2str(xmax),' and xmin to ',num2str(xmin),' to compensate,']);                
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


%Initialize solutiona and potential matrices:
U     = zeros(size(X));
V     = zeros(size(X));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION FORMULATION       %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%Linear steady-state exp profile
V    = (X.^2+Y.^2)/a;
U    = exp(-(X.^2+Y.^2)/(2*a));
%Compute stability bounds for time step k:
if(k==0)  
   hmin = h;  
   if(method==1 || method==11)  
      klin = hmin^2/(2*sqrt(2)*a);
   elseif((method==2 || method==12) || length(methodv)>1)
      klin = (3/4)*hmin^2/(2*sqrt(2)*a);
   end
   k = 0.8*klin;    
   disp(['Time step computed!  k: ',num2str(k)]);
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

%Display simulation info:
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
disp(['h:          ',num2str(h)]);
disp(['k:          ',num2str(k)]);
disp(['GridSize:   ',num2str(xres),'x',num2str(yres), ' = ',num2str(xres*yres)]);
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
end
if(BC==1)
    disp('Boundary Conditions:  Dirichlet');
elseif(BC==2)
    disp('Boundary Conditions:  MSD |U|^2 = B');
elseif(BC==3)
    disp('Boundary Conditions:  Uxx+Uyy = 0');
end
if(cuda == 1)
    disp( 'CUDA Parameters and Info:');
    disp(['BlockSize:     ',num2str(cudablocksizex),'x',num2str(cudablocksizey)]);
    disp(['CUDAGridSize:  ',num2str(numcudablocksx),'x',num2str(numcudablocksy)]);
    disp(['NumBlocks:     ',num2str(numcudablocks)]); 
    disp(['Shared Memory/Block: ',num2str(sharedmemperblock),'KB']);
    disp(['TotalGPUMemReq:      ',num2str(yres*xres*(9 + 2*(method-1))*(4*precision)/1024),'KB']); 
end
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             PLOTTING SETUP                      %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%Find maximums for plot axis:
maxMod2U  = max(U(:).*conj(U(:)));
maxRealU  = max(real(U(:)));
plot_max  = max([maxMod2U maxRealU]);
fsize     = 25;  %Set font size for figures.
fig_count = 1;

fig_mod2  = figure(fig_count);
fig_count = fig_count+1;
set(fig_mod2, 'Name','2D NLSE MOD2 t = 0','NumberTitle','off','Color','w');

%Plot mod-squared initial condition:
plot_mod2  = surf(X,Y,U.*conj(U), 'EdgeColor', 'none', 'EdgeAlpha', 1);
shading interp;
colormap(bone(1024));
axis([xmin xmax ymin ymax 0 (maxMod2U+0.1*maxMod2U)]);
set(gca,'DataAspectRatio',[1 1 2*(2*maxMod2U)/(xmax-xmin)]);
xlabel('x','FontSize',fsize); ylabel('y','FontSize',fsize); zlabel('\Psi','FontSize',fsize);
set(gca,'Fontsize',fsize);
caxis([0 maxMod2U+0.1*maxMod2U]);
axis([xmin xmax ymin ymax -(plot_max+0.1*plot_max) (plot_max+0.1*plot_max)]);
colormap(bone(500));
hold on;
plot_real  = surf(X,Y,real(U));
plot_imag  = surf(X,Y,imag(U));
hold off;
set(plot_real, 'FaceColor', 'none','Meshstyle','row',   ...
      'EdgeColor', 'b'  , 'EdgeAlpha', 0.1,'LineWidth',1);
set(plot_imag, 'FaceColor', 'none','Meshstyle','column',...
      'EdgeColor', 'r'  , 'EdgeAlpha', 0.1,'LineWidth',1);
axis vis3d;
view([-17 30]);
drawnow;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             BEGIN SIMULATION                    %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
n      = 0;
for chunk_count = 1:numframes

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
            U = NLSE2D_TAKE_STEPS_2SHOC(U,V,s,a,h^2,BC,chunk_size,k);
        else
            U = NLSE2D_TAKE_STEPS_2SHOC_CUDA_D(U,V,s,a,h^2,BC,chunk_size,k);
        end
     end
   end
   n = n + chunk_size;

%Detect blow-up:
if(max(real(U(:))) > tol || sum(isnan(U(:))) > 0)
  disp('CRAAAAAAAAAAAAAAAASSSSSSSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHH');
  break;
end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  PLOT SOLUTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(fig_mod2, 'Name',['2D NLSE MOD2 t = ',num2str((n*k),'%.2f')]);
set (plot_mod2, 'Zdata', U.*conj(U));
set (plot_real, 'Zdata', real(U));
set (plot_imag, 'Zdata', imag(U));   
drawnow;  
end %chunk-count
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%         SIMULATION IS OVER                      %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
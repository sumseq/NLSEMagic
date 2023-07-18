%NLSEmagic3D
%Program to integrate the three-dimensional Nonlinear Shrödinger Equation:
%i*Ut + a*(Uxx+Uyy+Uzz) - V(x,y,z)*U + s*|U|^2*U = 0.
%
%2011 Ronald M Caplan
%Developed with support from the 
%Computational Science Research Center at 
%San Diego State University.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  VERSION:  012    1/8/2012 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Clear any previous run variables and close all figures:
clear mex; close all; clear all;
%------------------Simulation paramaters----------------------
endtw       = 5;     %End time of simulation.
numframes   = 50;    %Number of desired frames to plot.
h           = 1/5;   %Spatial grid spacing 
k           = 0;     %Time step [Set to 0 to auto-compute smallest stable timestep]
method      = 1;     %Method:  1: CD O(4,2), 2: 2SHOC O(4,4)
cuda        = 1;     %Try to use CUDA code (if installed and compiled)
precision   = 2;     %Single (1) or double (2) precision.
tol         = 10;    %Simulation mod-squared tolerance to detect blowup

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION PARAMETERS        %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
s  = 0;
a  = 1; 
BC = 1;
%Compute numerical domain to be big enough to avoid BC effects:
xmax =  0.4*sqrt(-log(sqrt(eps))*2*a);
xmin = -xmax;      xmax = xmax;    
ymin = -xmax-1;    ymax = xmax+1;
zmin = -xmax-1.5;  zmax = xmax+1.5;
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('----------------------------------------------------');
disp('--------------------NLSEmagic3D---------------------');
disp('----------------------------------------------------');

%Adjust ranges so they are in exact units of h:
xmin = xmin - rem(xmin,h); ymin = ymin - rem(ymin,h); zmin = zmin - rem(zmin,h);
xmax = xmax - rem(xmax,h); ymax = ymax - rem(ymax,h); zmax = zmax - rem(zmax,h);

%Set up grid:    
xvec  = xmin:h:xmax;  xres  = length(xvec);
yvec  = ymin:h:ymax;  yres  = length(yvec);
zvec  = zmin:h:zmax;  zres  = length(zvec);
%Need meshgrid matrices to make initial conditions:
[X,Y,Z] = meshgrid(xvec,yvec,zvec);
[M,N,L] = size(X);
%Note: meshgrid transposes x and y.

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
            if(M - cudablocksizex*(numcudablocksx-1) == 1)
                disp('MSD CUDA ERROR: M (yres) is one cell greater than CUDA block x-direction,')
                ymax = ymax-h;   
                ymin = ymin+h;
                disp(['adjusting ymax to ',num2str(ymax),' and ymin to ',num2str(ymin),' to compensate,']);                
            end
            if(N - cudablocksizey*(numcudablocksy-1) == 1) 
                disp('MSD CUDA ERROR: N (xres) is one cell greater than CUDA block y-direction')
                xmax = xmax-h;     
                xmin = xmin+h;
                disp(['adjusting xmax to ',num2str(xmax),' and xmin to ',num2str(xmin),' to compensate,']);                   
            end    
            if(L - cudablocksizez*(numcudablocksz-1) == 1) 
                disp('MSD CUDA ERROR: L (zres) is one cell greater than CUDA block z-direction')
                zmax = zmax-h;           
                zmin = zmin+h;
                disp(['adjusting zmax to ',num2str(zmax),' and zmin to ',num2str(zmin),' to compensate,']);                  
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

%Initialize solutiona and potential matrices:
U     = zeros(size(X));
V     = zeros(size(X)); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%             INITIAL CONDITION FORMULATION       %%%%%%%%%
%%%%%%%%%                                                 %%%%%%%%%
%Linear steady-state exp profile with a "kick"
V     = (X.^2 + Y.^2 + Z.^2)/a;
U     = exp(-(X.^2 + Y.^2 + Z.^2)/(2*a)).*exp(-1i*0.5.*X);

if(k==0)  
   hmin = h;  
   if(method==1 || method==11)  
      klin = hmin^2/(3*sqrt(2)*a);
   elseif((method==2 || method==12) || length(methodv)>1)
      klin = (3/4)*hmin^2/(3*sqrt(2)*a);
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
fsize     = 25;  %Set font size for figures.
fig_count = 1;

xticklabels    = floor( (xmin + 1 +(xmax-1-xmin).*[0 1/4 1/2 3/4 1]));
yticklabels    = floor( (ymin + 1 +(ymax-1-ymin).*[0 1/4 1/2 3/4 1]));
zticklabels    = floor( (zmin + 1 +(zmax-1-zmin).*[0 1/4 1/2 3/4 1]));  
xticklocations = (xticklabels - xmin)./h;
yticklocations = (yticklabels - ymin)./h;
zticklocations = (zticklabels - zmin)./h;

fig_cube_mod2  = figure(fig_count);
fig_count      = fig_count+1;
set(fig_cube_mod2, 'Name','3D NLSE MOD2 t = 0','NumberTitle','off',...
              'InvertHardcopy','off','Color','k');

%Plot mod-squared initial condition:      
plot_cube_mod2 = vol3d('cdata',U.*conj(U),'texture','3D');
      
colormap(hsv(512));      
xlim([0 xres]); ylim([0 yres]); zlim([0 zres]);      
set(gca,'XTick',xticklocations,'XTickLabel',xticklabels);
set(gca,'YTick',yticklocations,'YTickLabel',yticklabels);
set(gca,'ZTick',zticklocations,'ZTickLabel',zticklabels);
set(gca,'DataAspectRatio',[xres/abs(xmax-xmin) yres/abs(ymax-ymin) zres/abs(zmax-zmin)]); 
grid on;
xlabel('x','FontSize',fsize); ylabel('y','FontSize',fsize); zlabel('z','FontSize',fsize);
set(gca,'Color','k','XColor','w','YColor','w','ZColor','w','FontSize',14);          

axis vis3d;
view([-29 9]);      
cmax_init = max(U(:).*conj(U(:)));
caxis([0 cmax_init]);
colorbar;
zoom(0.75)

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
   alphamap('rampup'); 
   alim([0 cmax_init]);  
end   
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
   end
   n = n + chunk_size;   
   
%Detect blow-up:
if(max(real(U(:))) > tol || sum(isnan(U(:))) > 0)
  disp('CRAAAAAAAAAAAAAAAASSSSSSSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHH');
  break;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  PLOT SOLUTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_cube_mod2.cdata = U.*conj(U); %Compute density
plot_cube_mod2       = vol3d(plot_cube_mod2);
set(fig_cube_mod2, 'Name',['3D NLSE  t = ',num2str((n*k),'%.2f')]);        
drawnow;     
end %chunk-count
%%%%%%%%%                                                 %%%%%%%%%
%%%%%%%%%         SIMULATION IS OVER                      %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

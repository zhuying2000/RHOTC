%%
clear;clc;
addpath(genpath(cd));
randn('seed',2013);
randn('seed',2013);
%%  Face data 真实数据
load('YaleFace.mat');
X = YaleFace./max(YaleFace(:)); 
[n1, n2 ,n3,n4] = size(X);
maxP = max(abs(X(:)));
maxP1 = max(YaleFace(:));
Nways=size(X);
    
%% initial parameters
opts.mu = 1e-4;
opts.max_mu = 1e10;
opts.tol = 1e-10; 
opts.rho = 1.1;
opts.DEBUG = 1;
opts.max_iter =500;
lambda = 1/sqrt(n3*n4*max(n1,n2));
%% 加入噪声 随机位置的噪声
p = 0.05;
m = p*n1*n2*n3*n4;
temp = rand(n1*n2*n3*n4,1);
[B,I] = sort(temp);
I = I(1:m);
Omega = zeros(n1,n2,n3,n4);
Omega(I) = 1;
E = sign(rand(n1,n2,n3,n4)-0.5);
S = Omega.*E; 
XT=X+S;
%% 
sr = 0.7;
fprintf('Sampling ratio = %0.8e\n',sr);
temp = randperm(prod(Nways));
kks = round((sr)*prod(Nways));
mark = zeros((Nways));
mark(temp(1:kks)) = 1;
ZZ_=XT.*mark; 

%%  
  fprintf('===== t-SVD by Discrete Cosine Transform =====\n');
   tic
     [Xhat,Shat ,~,trank] = HTNN_FFT(XT,mark,opts,lambda);  
   toc
  
RSE_x = norm(X(:)-Xhat(:))/norm(X(:));
fprintf('\nsampling rate: %f\n', sr);
fprintf('tubal rank of the recovered tensor: %d\n', trank);
fprintf('X relative recovery error: %.4e\n', RSE_x);









clear all
addpath(genpath(cd));
randn('seed',2013);
randn('seed',2013);

n  = 40;
n1 = n;
n2 = n;
n3 = n;
n4 = n;
r = floor(n*0.1); 
A=randn(n1,r,n3,n4)/n;
B=randn(r,n2,n3,n4)/n;
X = htprod_fft(A,B);
[n1, n2, n3, n4]=size(X);
maxP=max(X(:));
Nways=size(X);
%% 加入噪声 随机位置的噪声
p = 0.2;
m = p*n1*n2*n3*n4;
temp = rand(n1*n2*n3*n4,1);
[B,I] = sort(temp);
I = I(1:m);
Omega = zeros(n1,n2,n3,n4);
Omega(I) = 1;
E = sign(rand(n1,n2,n3,n4)-0.5);
S = Omega.*E; 
XS=X+S;

%%  
sr = 0.8;
fprintf('Sampling ratio = %0.8e\n',sr);
temp = randperm(prod(Nways));
kks = round((sr)*prod(Nways));
mark = zeros((Nways)); 
mark(temp(1:kks)) = 1;  
ZZ_=XS.*mark; 
%%  our methods 

opts.mu = 1e-4;
opts.max_mu = 1e10; 
opts.tol = 1e-10; 
opts.rho = 1.1;
opts.DEBUG = 1;
opts.max_iter =500;
lambda = 1/sqrt(n3*n4*max(n1,n2));
tic
[X1,~ ,~,tsvd_rank] = HTNN_FFT(XS,mark,opts,lambda);  
time1=toc;

RSE_X1 = norm(X(:)-X1(:))/norm(X(:));

function [L,E,obj,tsvd_rank] =HTNN_FFT(M,Omega,opts,lambda)



mu = 1e-3;
max_mu = 1e8;
tol = 1e-6; 
max_iter = 500;
rho = 1.2;
DEBUG = 1;

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end 
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end


dim = size(M);
p = length(size(M));
n = zeros(1,p);
for i = 1:p
    n(i) = size(M,i);
end

BarOmega = ones(dim) - Omega; 
%% 初始化  
E = zeros(dim);
Y1 = E;
Y2 = E;
L = E;
%% 使用ADMM算法 开始迭代  
for iter = 1 : max_iter
    Lk = L;
    Ek = E;
     
    %% updata Z   
     Z1bar= (M-E+L-(Y2-Y1)/mu)/2;
     Z2bar=L+Y1/mu;
     Zbar = Z1bar.*Omega + Z2bar.*BarOmega;
    
    %% update X     
    [L,tnnX,trank] =prox_htnn_F(Zbar-Y1/mu,1/mu); 
    
    %% updata Z 
     Z1= (M-E+L-(Y2-Y1)/mu)/2;
     Z2=L+Y1/mu;
     Z = Z1.*Omega + Z2.*BarOmega;
    
   %% update E  
    E = prox_l1((M-Z-Y2/mu),lambda/mu).*Omega; 
     dY1 =L- Z;
     dY2=(E-M+Z).*Omega;
     chgX = max(abs(Lk(:)-L(:)));
     chgE = max(abs(Ek(:)-E(:)));
     chg = max([chgX chgE max(abs(dY1(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0  
            obj = tnnX; 
            err = chg;
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err) ', errX=' num2str(chgX) ', errE=' num2str(chgE) ', errL_Z=' num2str(max(abs(dY1(:)))) ', errE+Z_M=' num2str(max(abs(dY2(:)))) ]); 
        end
    end
    
    if chg < tol 
        break;
    end 
    Y1 = Y1 + mu*dY1; 
    Y2=Y2+ mu*dY2;
    mu = min(rho*mu,max_mu);   
end
obj = tnnX;
tsvd_rank=trank;
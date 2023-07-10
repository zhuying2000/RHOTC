function [S,iperm] = tmprod(T,U,mode,transpose,saveperm)
%TMPROD Mode-n tensor-matrix product.
%% 一种参数输入情况 只求n模积 向量mode里面的参数应该不一样
%   S = tmprod(T,U,mode) computes the tensor-matrix product of the tensor T
%   with the matrices U{1}, ..., U{N} along the modes mode(1), ...,
%   mode(N), respectively. Note that in this implementation, the vector
%   mode should contain distinct integers. The mode-n tensor-matrix
%   products are computed sequentially in a heuristically determined order.
%   A mode-n tensor-matrix product results in a new tensor S in which the
%   mode-n vectors of a given tensor T are premultiplied by a given matrix
%   U{n}, i.e., tens2mat(S,mode(n)) = U{n}*tens2mat(T,mode(n)).
%% 一种参数输入情况 求n模积 对U进行共轭转置
%   S = tmprod(T,U,mode,'T') and tmprod(T,U,mode,'H') apply the mode-n
%   tensor-matrix product using the transposed matrices U{n}.' and
%   conjugate transposed matrices U{n}' respectively along mode(n).
%% 一种参数输入情况 求n模积 保存一种置换操作 n模积可以通过置换恢复
%   [S,iperm] = tmprod(T,U,mode) and S = tmprod(T,U,mode,'saveperm') save
%   one permutation operation. In the former case, the tensor-matrix
%   product can then be recovered by permute(S,iperm).
%
%   See also tens2mat, mat2tens, contract.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be),
%            Nick Vannieuwenhoven (Nick.Vannieuwenhoven@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)

% Check arguments.
%% 判断输入参数 看是前面解释的哪一种情况
if nargin < 4, transpose = 0; end  %不进行对U的共轭转置
if nargin < 5, saveperm = false; end %不进行保存恢复操作
if ischar(saveperm), saveperm = strcmpi(saveperm,'saveperm'); end %ischar是否为数组 strcmpi字符串大小比较 同1 否0
switch transpose
    case {'T','H'}, m = [2 1];%如果输入字符为 T 或者H 说明要进行共轭转置操作
    otherwise, m = [1 2]; %不进行共轭转置操作
        if nargin < 5, saveperm = transpose; end  %你干啥呢
end
%% 判断可不可以进行n模积操作 有点疑惑细看 难难难 下面回看
if ~iscell(U), U = {U}; end %传进来的U是U{1}或者U{2} 是一个矩阵 所以这里判断是不是元胞数组 把输入进来的矩阵变成元胞数组？
if length(U) ~= length(mode)%n模积  要把张量按n模展开为矩阵，再与U相乘 。疑惑length(mode)应该一直为1呀 U是矩阵 length(U)肯定不为1。解惑了 上一步把U变为了元胞数组 所以其长度其实也为1
    error('tmprod:NumberOfProducts','length(U) should be length(mode).');
end
U = U(:)'; mode = mode(:)';%你不是常数吗？你干啥 或者在原本代码中mode不是常数 被本文作者拿来用了？
size_tens = ones(1,max(mode));%就是初始化size_tens让它去装T的各个维度的大小，你整这么复杂干啥 有深意？
size_tens(1:ndims(T)) = size(T);%ndims 返回数组的维数，维数总是大于等于2 。如果维数为1，那么它返回的是2 奇怪当mode为3时size_tens 初始化值为 [1 1 1]，但是后面size_tens(1:ndims(T))是size_tens(1:4) 大小都不一样还赋值，命令行窗口试了 这样虽然不会报错 但是非常奇怪             
if any(cellfun('size',U,m(2)) ~= size_tens(mode))%A = cellfun('size',C,k) 返回沿 C 每个元素的第 k 维的大小。这里的解释是返回沿U的第m（2）维的大小？
    error('tmprod:U','size(T,mode(n)) should be size(U{n},%i).',m(2));
end

% Sort the order of the mode-n products.
[~,idx] = sort(size_tens(mode)./cellfun('size',U,m(1)));%sort 从小到大 x = A./B A的每个除以B的每个
mode = mode(idx);
U = U(idx);

% Compute the complement of the set of modes.
n = length(mode);
N = length(size_tens);
bits = ones(1,N);
bits(mode) = 0;
modec = 1:N;
modec = modec(logical(bits(modec)));%logical将数值转换为逻辑值 非0为1 0为0

% Prepermute the tensor.
perm = [mode modec];
size_tens = size_tens(perm);
S = T; if any(mode ~= 1:n), S = permute(S,perm); end

% Cycle through the n-mode products.
for i = 1:n
    size_tens(1) = size(U{i},m(1));
    switch transpose
        case 'T'
            S = reshape(U{i}.'*reshape(S,size(S,1),[]),size_tens);
        case 'H'
            S = reshape(U{i}'*reshape(S,size(S,1),[]),size_tens);
        otherwise
            S = reshape(U{i}*reshape(S,size(S,1),[]),size_tens);
    end
    if i < n
        S = permute(S,[2:N 1]);
        size_tens = size_tens([2:N 1]);
    end
end

% Inverse permute the tensor, unless the user intends to do so himself.
iperm(perm([n:N 1:n-1])) = 1:N;
if nargout <= 1 && ~saveperm, S = permute(S,iperm); end








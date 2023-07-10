function I = unfoldi(i,j,L)
%阿巴巴 你在干什么
% Written by  Wenjin Qin  (qinwenjin2021@163.com)

I = ones(1,j);
    for t = j:-1:3   %注意这里当j的值为3时 它也会跑一次
        I(t) = ceil(i/L(t-1)); %ceil 向上取整  
        i = i-(I(t)-1)*L(t-1);
    end
end

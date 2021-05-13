dgt = zeros(10,10);
dgt(2:9,2:9) = 1;
g = logical(dgt);

% resmap
res = zeros(10,10);
res(3:8,3:8) = 1;

E = abs(res - dgt);

[d, idx] = bwdist(dgt);
K = fspecial('gaussian',3,1);
Et = E;
Et(~g) = Et(idx(~g));
EA = imfilter(Et,K);
MIN_E_EA = E;
MIN_E_EA(g & EA<E) = EA(g & EA<E);
%
B = ones(size(g));
B(~g) = 2.0-1*exp(log(1-0.5)/5.*d(~g));
Ew = MIN_E_EA.*B;

TPw = sum(dgt(:)) - sum(sum(Ew(g)));
FPw = sum(sum(Ew(~g)));

R = 1-mean2(Ew(g));
P = TPw./(eps+TPw+FPw);
Q = (2)*(R*P)./(eps+R+P);

TP = sum(dgt(:)) - sum(sum(E(g)));
FP = sum(sum(E(~g)));


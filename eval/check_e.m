g = zeros(10,10);
g(2:9,3:7) = 1;
g = logical(g);

% resmap
res = zeros(10,10);
res(2:8,3:9) = 1;
res = logical(res);

dg = double(g);
dres = double(res);

if (sum(dg(:)) == 0)
    e_matrix = 1.0 - dres;
elseif(sum(~dg(:)) == 0)
    e_matrix = dres;
else
    mu_res = mean2(dres);
    mu_g = mean2(dg);
    align_res = dres - mu_res;
    align_g = dg - mu_g;

    align_matrix = 2.*(align_g.*align_res)./(align_g.*align_g + align_res.*align_res + eps);
    e_matrix = ((align_matrix + 1).^2)/4;
end

[w,h] = size(g);
score = sum(e_matrix(:))./(w*h - 1 + eps);

% function [align_matrix] = Align(dres, dg)
% mu_res = mean2(dres);
% mu_g = mean2(dg);
% align_res = dres - mu_res;
% align_g = dg - mu_g;

% align_matrix = 2.*(align_g.*align_res)./(align_g.*align_g + align_res.*align_res + eps);
% end

% function enhanced = EAlign(align_matrix)
% enhanced = ((align_matrix + 1).^2)/4;
% end
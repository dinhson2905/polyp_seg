map = ones(5,5);
target = zeros(5,5);
threshold = 0.5;

label = zeros(size(target));
label(map >= threshold) = 1;
numRec = length(find(label3 == 1));

[a, b, c, d, e, f] = Fmeasure_calu(map,double(target),size(target),threshold);
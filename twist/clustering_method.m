function [result,labelnew]=clustering_method(W,nClass,gndnew)
% returen the results of SC
% reture the results of k-means

D=sum(W,2);
DD=D.^(-0.5);
L=diag(DD)*W*diag(DD);
[V,~] = eig(L);
H = V(:,1:nClass);
Q=normr(H);
labelnew = litekmeans(Q,nClass,'Replicates',20);

result = Clustering8Measure(labelnew,gndnew);
disp('sc finished')


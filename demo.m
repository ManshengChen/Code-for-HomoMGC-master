clear
addpath(genpath('.\'))

load('ACM3025.mat');

n = size(feature,1); % datasize
I = eye(n);
[~, gt] = max(label, [], 2);
nClass=max(gt);

%%
%ACM
r1 = cal_homo_ratio(PAP, gt, true);
r2 = cal_homo_ratio(PLP, gt, true);
fprintf('The homophily ratios of these two view are:%12.6f %12.6f\n',[r1 r2]);

%%
SX = constructW_PKN(feature', 5); %the range of k {1,5,10,15,20,25}
SX(SX~=0) = 1;
SX = SX+I;

%ACM
A{1} = PAP;
A{2} = PLP;

lambda = 0.001; %the range of lambda {0.0001,0.001,0.01,0.1,1,2}


%%
CA=(A{1}+A{2})/2;
[A,E,B]=TensorGraph(SX,CA,lambda);

%% Fianl partition
S = A(:,:,2)+A(:,:,2)';
[result,pre_gt]=clustering_method(S,nClass,gt);
save('ACM3025_results.mat','gt','pre_gt'); %For fairness, the same evaluation method as baselines is used.

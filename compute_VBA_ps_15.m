%RUN VBA Bayesian model selection random effect analysis for the middle
%layer FE

toolbox_path = '/home/membercophy/Documents/MATLAB/VBA-toolbox-master';
addpath(genpath(toolbox_path));

original_dir = pwd;
cd(toolbox_path);
VBA_setup();
cd(original_dir);

%%

patch_sizes = ["1","25", "5", "75", "10"];
load(sprintf('/home/membercophy/Matteo_M/V1_ERF/V1_ERF_F_diff/FE_mid_lay/V1_avg_FE_mid_lay_%s.mat', patch_sizes(1)))
F_matrix = [];
for i = 1:length(patch_sizes)
    patch = patch_sizes(i);
    load(sprintf('/home/membercophy/Matteo_M/V1_ERF/V1_ERF_F_diff/FE_mid_lay/V1_avg_FE_mid_lay_%s.mat', patch))
    F_matrix = [F_matrix; FE_mid_lay(:).'];
end 

options = struct();
options.niter = 100;
options.DisplayWin = 0; 
options.families = {[1],[2],[3],[4],[5]}; % families -patches

famEP = zeros(1, 5);
PEP  = zeros(1, 5);

[posterior, out] = VBA_groupBMC(F_matrix, options);

% Bayesian Omnibus Risk
p_H0 = out.bor; %null: p(H0|y) 
fprintf('BOR %.9f\n',p_H0)

% Protected Exceedance Probabilities
K = length(out.families.ep); % 5families 
EP = out.ep; 
famEP = out.families.ep; % EP for 5 families
PEP = (1 - out.bor) .* out.families.ep + out.bor / K; % PEPs for family bs our out.bor is against 5 families

[~, bestPatch] = max(PEP); 
fprintf("Best patch %f\n",bestPatch)
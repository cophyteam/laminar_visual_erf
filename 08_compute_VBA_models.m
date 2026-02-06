


load('aligned_data_ERF_F.mat')
% RUN VBA Bayesian modle selection random effect analysis 
F_con = cat(1, la_aligned_rh_layer_F_diff, ra_aligned_lh_layer_F_diff);
F_ips = cat(1, la_aligned_lh_layer_F_diff, ra_aligned_rh_layer_F_diff);


F = F_con % or F_ips for other condition

%F 40*6*360 subjects ()*layers*time

% ========================
% PARAMETERS
% ========================
dt = (0.3 - (-0.3)) / (360 - 1);  
time = linspace(-0.3, 0.3, 360);

options = struct();
options.niter = 100
options.DisplayWin = 0; 
options.families = {[1, 2, 3], [4], [5, 6]} ;
nSubjects   = size(F,1);   % 40 in ERF
nModels     = size(F,2);   % 6 layers
nTimepoints = size(F,3);   % times

EP  = zeros(6, nTimepoints);
famEP = zeros(3, nTimepoints);
p_H0 = zeros(1, nTimepoints); % BOR in time
PEP  = zeros(3, nTimepoints);
a_model  = zeros(6, nTimepoints);


for t = 1:nTimepoints
    
    % models × subjects
    F_t = squeeze(F(:,:,t)).';
    
    [posterior, out] = VBA_groupBMC(F_t, options);
    
    % Bayesian Omnibus
    p_H0(t) = out.bor; %null: p(H0|y) 
    
    % Protected Exceedance Probabilities
    K = length(out.families.ep); %3 family 
    EP(:,t) = out.ep; %ep for 6 layers 
    famEP(:,t) = out.families.ep; %ep for 3 families
    PEP(:,t) = (1 - out.bor) .* out.families.ep + out.bor / K; %pep for family bs our out.bor is against 3 family
    
end

bestModelOverTime = zeros(1, nTimepoints);
for t = 1:nTimepoints
    
    if p_H0(t) < 0.5 

        [~, bestModel] = max(PEP(:,t)); 
        bestModelOverTime(t) = bestModel;

    end
    
end

save('ERF_OverTime_Contra_family.mat', 'bestModelOverTime', 'time', 'p_H0', 'EP', 'famEP');

figure;
plot(time, p_H0, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('p(H0|y)');
title('Bayesian Omnibus Risk over Time');
grid on;

figure;
imagesc(time, 1:3, PEP);
set(gca,'YDir','normal');
colorbar;
xlabel('Time (s)');
ylabel('Model');
title('Protected Exceedance Probabilities (PEP)');







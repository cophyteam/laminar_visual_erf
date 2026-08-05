%RUN VBA Bayesian model selection random effect analysis 

toolbox_path = '/home/membercophy/Documents/MATLAB/VBA-toolbox-master';
addpath(genpath(toolbox_path));

original_dir = pwd;
cd(toolbox_path);
VBA_setup();
cd(original_dir);

%%
patch_sizes = [25, 75, 10];
window_sizes = [25, 50];

for ws = 1:length(window_sizes)
    window_size = window_sizes(ws);
    for ps = 1:length(patch_sizes)
        patch_size = patch_sizes(ps);
        
        filename = sprintf('/home/membercophy/Matteo_M/V1_ERF/V1_ERF_F_diff/ws%d/ps%d/aligned_V1_ERF_FE_ws%d_ps%d.mat', ...
                           window_size, patch_size, window_size, patch_size);

        load(filename);
        data_struct.F_contra = cat(1, la_aligned_rh_layer_F_diff, ra_aligned_lh_layer_F_diff);
        data_struct.F_ipsi   = cat(1, la_aligned_lh_layer_F_diff, ra_aligned_rh_layer_F_diff);

        F_conditions = ["F_ipsi", "F_contra"];

        for i = 1:length(F_conditions)
            condition_name = F_conditions(i);

            F_matrix = data_struct.(condition_name);

            % ========================
            % PARAMETERS
            % ========================
            dt = (0.3 - (-0.3)) / (360 - 1);  
            time = linspace(-0.3, 0.3, 360);

            options = struct();
            options.niter = 100;
            options.DisplayWin = 0; 
            options.families = {[1, 2, 3], [4], [5, 6]} ;
            nTimepoints = size(F_matrix,3);   % times

            EP  = zeros(6, nTimepoints);
            famEP = zeros(3, nTimepoints);
            p_H0 = zeros(1, nTimepoints); % BOR in time
            PEP  = zeros(3, nTimepoints);

            for t = 1:nTimepoints

                % models × subjects
                F_t = squeeze(F_matrix(:,:,t)).';

                [posterior, out] = VBA_groupBMC(F_t, options);
                % Bayesian Omnibus
                p_H0(t) = out.bor; %null: p(H0|y) 

                % Protected Exceedance Probabilities
                K = length(out.families.ep); %3 families 
                EP(:,t) = out.ep; % EP for 6 layers 
                famEP(:,t) = out.families.ep; % EP for 3 families
                PEP(:,t) = (1 - out.bor) .* out.families.ep + out.bor / K; % PEPs for family bs our out.bor is against 3 family

            end

            bestModelOverTime = zeros(1, nTimepoints);
            for t = 1:nTimepoints

                if p_H0(t) < 0.5 

                    [~, bestModel] = max(PEP(:,t)); 
                    bestModelOverTime(t) = bestModel;

                end

            end

            filename = sprintf('/home/membercophy/Matteo_M/V1_ERF/V1_ERF_F_diff/ERF_OverTime_%s_family_ws%d_ps%d.mat', condition_name, window_size, patch_size);
            save(filename, 'bestModelOverTime', 'time', 'p_H0', 'EP', 'famEP', 'PEP');

            figure;
            plot(time, p_H0, 'LineWidth', 2);
            xlabel('Time (s)');
            ylabel('p(H0|y)');
            title(sprintf('Bayesian Omnibus Risk over Time - %s', condition_name));
            grid on;

            figure;
            imagesc(time, 1:3, PEP);
            set(gca,'YDir','normal');
            colorbar;
            xlabel('Time (s)');
            ylabel('Model');
            title(sprintf('Protected Exceedance Probabilities (PEP) - %s', condition_name));

        end 
    end 
end



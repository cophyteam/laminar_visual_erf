% === INITIAL SETUP ===
restoredefaultpath;
rehash toolboxcache;

% === ADD SPM PATH ===
addpath('/spm_25.01.02/spm');
spm('defaults','eeg');
warning('off','MATLAB:dispatcher:nameNonexistent'); % suppress missing subfolder warnings

% === USER-DEFINED PARAMETERS ===
subj_list     = {'105', '107', '109', '113', '114','115', '117','118','119', '123', '124', '126','127','129', '130','131','132','133','134','135','136', '137','138', '141','142','143','144','145'};   % subjects to process
apply_crop     = 0;              % crop epochs
apply_baseline = 0;              % apply baseline correction
apply_filter   = 0;              % apply filters

crop_time      = [-300 300];     % ms
baseline_time  = [-500 -50];       % ms
filter_band    = [1 40];         % Hz

sides = {'right','left'};        % process both sides
base_dir = '';



% === MAIN LOOP ===
for s = 1:length(subj_list)
    subj = subj_list{s};
    fprintf('\n=== Processing subject %s ===\n', subj);


for side_idx = 1:length(sides)
    side = sides{side_idx};
    fprintf('\n--- Side: %s ---\n', side);

    meg_dir = fullfile(base_dir, ['sub-' subj], 'ses-01', 'meg');

    % 
    ds_folders = dir(fullfile(meg_dir, ['sub-' subj '_ses-01_task-V100V100_acq-*_meg.ds']));

    acq_nums = zeros(1, numel(ds_folders));
    for i = 1:numel(ds_folders)
        tokens = regexp(ds_folders(i).name, 'acq-(\d+)_meg\.ds', 'tokens', 'once');
        if ~isempty(tokens)
            acq_nums(i) = str2double(tokens{1});
       
        end
    end

    [~, max_idx] = max(acq_nums);
    best_ds = ds_folders(max_idx).name;

    fprintf('Using dataset folder: %s\n', best_ds);

    % 
    fname = fullfile(meg_dir, best_ds, ...
    'mxf_no_mxf_data', ...
    'mxf_sess_data', ...
    'spm_convert_epo', ...
    ['spm_sub-' subj '_ses-01_task-V100V100_meg_no_mxf_' side '_stim_nrg_ERF_epo.mat']);

   


    if ~isfile(fname)
        fprintf('File not found for %s %s — skipping.\n', subj, side);
    end

    % === LOAD DATA ===
    load(fname);

    % === CLEAN UP LABELS ===
    labels = {D.trials.label};
    if numel(unique(labels)) > 1
        for t = 1:length(D.trials)
            D.trials(t).label = 'Undefined';
        end
    end
    save(fname,'D');

    % === INITIALIZE JOBMAN ===
    spm_jobman('initcfg');
    matlabbatch = {};
    job_idx = 1;

    % === OPTIONAL PREPROCESSING ===
    if apply_crop
        fprintf('Adding cropping: [%d %d] ms\n', crop_time(1), crop_time(2));
        matlabbatch{job_idx}.spm.meeg.preproc.crop.D = {fname};
        matlabbatch{job_idx}.spm.meeg.preproc.crop.timewin = crop_time;
        job_idx = job_idx + 1;
    end

    if apply_baseline
        fprintf('Adding baseline correction: [%d %d] ms\n', baseline_time(1), baseline_time(2));
        matlabbatch{job_idx}.spm.meeg.preproc.bc.D = {fname};
        matlabbatch{job_idx}.spm.meeg.preproc.bc.timewin = baseline_time;
        job_idx = job_idx + 1;
    end

    if apply_filter
        fprintf('Adding filter: [%d %d] Hz band-pass\n', filter_band(1), filter_band(2));
        matlabbatch{job_idx}.spm.meeg.preproc.filter.D = {fname};
        matlabbatch{job_idx}.spm.meeg.preproc.filter.type = 'butterworth';
        matlabbatch{job_idx}.spm.meeg.preproc.filter.band = 'bandpass';
        matlabbatch{job_idx}.spm.meeg.preproc.filter.freq = filter_band;
        matlabbatch{job_idx}.spm.meeg.preproc.filter.dir = 'twopass';
        matlabbatch{job_idx}.spm.meeg.preproc.filter.order = 5;
        job_idx = job_idx + 1;
    end

    % === AVERAGING STEP ===


    % [filepath, filename, ext] = fileparts(fname);
    % baseline_file = fullfile(filepath, ['b' filename ext]);

    % fprintf('Adding averaging step...\n');
    % matlabbatch{job_idx}.spm.meeg.averaging.average.D = {baseline_file};
    % matlabbatch{job_idx}.spm.meeg.averaging.average.userobust.standard = false;
    % matlabbatch{job_idx}.spm.meeg.averaging.average.plv = false;
    % matlabbatch{job_idx}.spm.meeg.averaging.average.prefix = 'bm';

    fprintf('Adding averaging step...\n');
    matlabbatch{job_idx}.spm.meeg.averaging.average.D = {fname};
    matlabbatch{job_idx}.spm.meeg.averaging.average.userobust.standard = false;
    matlabbatch{job_idx}.spm.meeg.averaging.average.plv = false;
    matlabbatch{job_idx}.spm.meeg.averaging.average.prefix = 'm';

    % === RUN THE PIPELINE ===
    spm_jobman('run', matlabbatch);

    % === OUTPUT INFO ===
    [output_dir, output_file, ~] = fileparts(fname);
    fprintf('\nProcessing completed for %s (%s side).\n', subj, side);
    fprintf('Averaged file saved as:\n%s/m%s.mat\n', output_dir, output_file);
end


end

fprintf('\n=== All processing finished successfully! ===\n');
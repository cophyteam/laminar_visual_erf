
import mne
import numpy as np
import os 
import glob
import time
import matplotlib.pyplot as plt 
import pandas as pd 
import autoreject


def upload_file(rawdata_path, sub, ses, condition):
    """
    Search and upload the raw datachunck

    :param rawdata_path: datapath
    :param sub: subjects id
    :param ses: session id 
    :param condition: experimental condition (block-wise)
    """

    
    sub_meg_path = os.path.join(rawdata_path, "sub-" + sub , "ses-0" + ses , "meg/sub-"+ sub + "_ses-0" + ses + "_task-" + condition + "_acq-")
    folders_acq = glob.glob(sub_meg_path + "*_meg.ds", recursive=True)
    list_folders = sorted(folders_acq, key=lambda x: int(x.split('acq-')[1].split('_')[0]))
    
    return list_folders


def bandpass_filter(raw_instance, lo_pass, hi_pass):
    """
    Apply bandpass filter to the raw .fif file   
    :param raw_instance: raw .fif file
    :param lo_pass: low frequency
    :param hi_pass: high frequency

    """
    raw_instance.plot_psd(tmin=0, tmax=60, fmin=0, fmax=300, average=True, spatial_colors=False)
    plt.show()
    filtered_raw = raw_instance.copy()
    filtered_raw.filter(hi_pass, lo_pass, method='fir', phase='zero') # non-causal filter, overlap-add filtering
    filtered_raw.plot_psd(tmin=0, tmax=60, fmin=0, fmax=300, average=True, spatial_colors=False)
    plt.show()

    return filtered_raw


def notch_filter(raw_instance):
    """
    Filter for line noise removal - fixed 50Hz
    
    :param raw_instance: raw .fif 
    """

    raw_instance.plot_psd(tmin=0, tmax=60, fmin=0, fmax=300, average=True, spatial_colors=False)
    notch_filtered_raw = raw_instance.copy()
    freqs = np.array([50]) # Power line at 50Hz 
    notch_filtered_raw.notch_filter(freqs)
    notch_filtered_raw.plot_psd(tmin=0, tmax=60, fmin=0, fmax=300, average=True, spatial_colors=False)
    plt.show()

    return notch_filtered_raw  


def downsampling(raw, new_freq, events_array):
    """
    Downsample raw file and resample events 
    
    :param raw: raw fifi file to be downsampled
    :param new_freq: downsampling frequency
    :param events_array: original event array to resample at the new frequency
    """

    if isinstance(new_freq, (int, float)):
        events_array, raw_resampled = raw.copy().resample(sfreq=new_freq, events= events_array)
        print(f"New sampling frequency {new_freq} Hz")
        return raw_resampled, events_array

    elif new_freq is None:
        print("No resampling applied.")
        return raw  
    else:
        print(f"Error: ({type(new_freq)}). No resampling.")
        return raw


def apply_ICA(raw_instance, n_comp):
    """
    Appply ICA and plot intermediate steps 
    
    :param raw_instance: raw .fif file
    :param n_comp: number of components
    """

    ICA_data = raw_instance.copy()
    
    ica = mne.preprocessing.ICA(n_components=n_comp, random_state=15, method='picard')
    ica.fit(ICA_data)
    for i in range(n_comp):
        explained_var_ratio = ica.get_explained_variance_ratio(ICA_data, components=[i])
        ratio_percent = round(100 * explained_var_ratio['mag'])
        print(
            f"Fraction of variance explained by component {i}: "
            f"{ratio_percent}%"
        )

    ica.plot_sources(ICA_data, show = True)
    plt.show()
    ica.plot_components()
    plt.show()
    time.sleep(1) 
    ics_to_plot = list(range(0,11))
    ica.plot_properties(ICA_data, picks = ics_to_plot)
    plt.show()

    exclude_input = input("Index components to exclude with comma:")
    
    if exclude_input:
        try:
            exclude_comp = np.array([int(comp.strip()) for comp in exclude_input.split(',')])
        except ValueError:
            print("Invalid input.")
    else:
        print("No components selected.")
        exclude_comp = None
        
    cleaned_raw = ica.apply(ICA_data, exclude=exclude_comp) 
    
    return cleaned_raw


def auto_bad_epochs(epochs): 
    """
²   Run automatic epoch rejections relyong of Autoreject algorithm -return list of bad epochs and repaired epochs
        
    :param epochs: epochs for automatic artifact removal 
    """
    
    picks_channels = mne.pick_types(epochs.info, meg=True, eeg=False,ref_meg=False,
                   misc=False, chpi=False)

    ar = autoreject.AutoReject(
        consensus=np.linspace(0, 1.0, 27),
        n_interpolate=np.array([1, 4, 32]),
        thresh_method="bayesian_optimization",
        cv=10,
        picks = picks_channels,
        n_jobs=-1,
        random_state=42,
        verbose="progressbar"
    )
    ar.fit(epochs) 
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    epochs[reject_log.bad_epochs].plot()
    
    repaired_epochs = epochs_ar
    
    bad_epochs_info = reject_log.bad_epochs
    
    return repaired_epochs, bad_epochs_info



def create_epochs_check_tables(tables_path, sub, ses, blocks, raw_instance, 
                               events_array,manual_removal, 
                               t_min, t_max, dict_events = True):
    
    """
    create the epochs, using the tables available for each session exclude the data segments whee blinks where detected, 
    with too fast RT or incorrect response

    :param tables_path: behavioral tables with RT; blinks and correcteness information for the block
    :param sub: subject id
    :param ses: session id
    :param blocks table id
    :param raw_instance: raw .fif file 
    :param events_array: event array resampled if necessary
    :param manual_removal: use the tables or not for artefact removal
    :param t_min: epoch start in s 
    :param t_max: epoch end in s
    :param dict_events: events id
    """

    resized_table = None
    epochs = None
    
    if dict_events is True: 

        tables_list = [os.path.join(tables_path, f"Session{ses}beh", f"Suj{sub}_ses-0{ses}_bloc_0{block}.csv") for block in blocks]
        
        combined_tables = pd.concat([pd.read_csv(table_path) for table_path in tables_list], axis=0)
        
        event_dict = { "left_stim_LE" : 3,
                        "left_stim_RE" : 19,
                        "left_stim_LD" : 7,
                        "left_stim_RD" : 23,
                        "right_stim_LE" : 35,
                        "right_stim_RE" : 51,
                        "right_stim_LD" : 39,
                        "right_stim_RD" : 55,
                        #"baseline_start": 252, # sometimes 252 / 255
                        #"left_resp" : 254,
                        #"right_resp": 253,
                        #"meg_start": 253,
                        #"baseline_1" :1,
                        #"baseline_2": 2,
                        }
        
        events_id_array = events_array[np.isin(events_array[:, 2], list(event_dict.values()))]
        
        subset_tables = combined_tables[combined_tables["trigger"].isin(events_id_array[:,2])]
        
        if len(subset_tables) == len(events_id_array):
            print("tables and event array coincide in lenght") # check that the lengh of epochs and tables match 
            to_filter_events_array = events_id_array
            tables_to_filter = subset_tables

        elif len(subset_tables) > len(events_id_array): #tables are longer than epochs available 
            print("More trials in behavioral tables then epochs - checking and resizing tables")
            mask = subset_tables[subset_tables["trigger"] == events_id_array[:,2]]
            resized_table = subset_tables[mask]
            tables_to_filter = resized_table
            print("Saving a single new table - matching the epochs number.")
            resized_table.to_csv(os.path.join(tables_path, f"Suj{sub}_ses-0{ses}_match_epo_blocks.csv"), index=True)
            
            to_filter_events_array = events_id_array
            tables_to_filter = resized_table

        elif len(subset_tables) < len(events_id_array): # more epochs than trials in tables 
            print("More epochs then trials in behav tables - redefining event array before epoch creation")
            mask = subset_tables[subset_tables["trigger"] == events_id_array[:,2]]
            resized_event_array = events_id_array[mask]

            to_filter_events_array = resized_event_array
            tables_to_filter = subset_tables

        else:
            print("Check events array and tables behavioral")
            
        if manual_removal is True:
            print("Creating epochs manually excluding bad trials")

            incorrect_rows = (tables_to_filter.iloc[:, 12] == 0) | (tables_to_filter.iloc[:, 16] != 0) | \
                 (tables_to_filter.iloc[:, 17] != 0) | (tables_to_filter.iloc[:, 13] < 0.250) # Wrong response, blink before, blink after stim onset, RT too low
                     
            events_array_filtered_manual = to_filter_events_array[~incorrect_rows]
            
            print(events_array_filtered_manual)
            
            print(f"Using the manual criteria - {len(to_filter_events_array) - len(events_array_filtered_manual)} epochs removed")
            
            epochs = mne.Epochs(raw_instance, events = events_array_filtered_manual, tmin= t_min, tmax= t_max, 
                                baseline= (None,0), 
                                event_id=event_dict, preload=True)
            
        elif manual_removal is None:
            
            epochs = mne.Epochs(raw_instance, events = events_id_array, tmin= t_min, 
                                tmax= t_max, baseline= (None, 0), 
                                event_id=event_dict, preload=True)

    elif dict_events is None:
        print("No dict events then all events will be epoched")
        event_dict = None # If None, all events will be used and a dict is created with string integer names corresponding to the event id integers.
        epochs = mne.Epochs(raw_instance, events = None, tmin = t_min, tmax =t_max, 
                            baseline= (None, 0), 
                            event_id=event_dict, preload=True)
    
    return epochs, resized_table



rawdata_path = ""
SUB = ['105', '107', '109', '113', '114',
           '115', '117','118','119', '123', 
           '124', '126','127','129', '130',
           '131','132','133','134','135', 
           '136', '137','138', '141','142',
           '143','144','145']
ses = "1"
condition = "V100V100"
mxf_option = "no_mxf"
tables_preproc_path = ""
behav_block = ["1", "2"]

for s, sub in enumerate(SUB): 
    
    list_folders = upload_file(rawdata_path, sub, ses, condition)
    
    no_mxf_path_dir = os.path.join(list_folders[-1], f"mxf_{mxf_option}_data", "mxf_sess_data") 
    no_mxf_raw_path = os.path.join(no_mxf_path_dir, f"sub-{sub}_ses-0{ses}_task-{condition}_meg_{mxf_option}.fif" )
    no_mxf_raw = mne.io.read_raw_fif(no_mxf_raw_path, preload=False, verbose=None)
    
    stim_chn = 'UPPT002'
    events = mne.find_events(no_mxf_raw, stim_channel= stim_chn)
    events_array, raw_downsamp = downsampling(no_mxf_raw, 600, events) # downsample fif MEG along with the events
    filtered_raw = bandpass_filter(raw_downsamp, lo_pass = 60, hi_pass = 0.1)
    
    filtered_target_fn = f"sub-{sub}_ses-0{ses}_task-{condition}_meg_{mxf_option}_filter.fif"
    save_output_path = os.path.join(no_mxf_path_dir, filtered_target_fn)
    notch_filtered_raw = notch_filter(filtered_raw)
    notch_filtered_raw.save(save_output_path, overwrite = True)

    notch_filtered_raw = mne.io.read_raw_fif(save_output_path, preload=False, verbose=None)

    " Do not apply ICA "
    #cleaned_raw = apply_ICA(no_mxf_path_dir, notch_filtered_data,  n_comp = 25, which_comp = ()) # apply ICA
    
    names_channels = notch_filtered_raw.ch_names
    chennels_to_excl = []
    for c, chan in enumerate(names_channels):
        if chan.startswith(("E","S","B","G","P","Q", "R")):
            chennels_to_excl.append(chan)
    print(f" excluded {len(chennels_to_excl)} channels from recording raw - no ref_meg, eeg")
    
    notch_filtered_nrg_raw = notch_filtered_raw.drop_channels(chennels_to_excl)

    defined_epochs, resized_tables = create_epochs_check_tables(tables_preproc_path, sub, ses, behav_block,
                                                                notch_filtered_nrg_raw, events_array, manual_removal=True,
                                                                t_min=-0.3, t_max=0.3, dict_events=True) 
    
    #inspect epochs
    defined_epochs.plot_image(picks="mag", combine="mean")
    defined_epochs[1].plot()
    defined_epochs.plot_image(picks="MRO51", combine="mean")
    
    "Do not apply Aftereject automatic bad epochs rejection"
    repaired_epochs, bad_epochs_info = auto_bad_epochs(defined_epochs)
    good_epochs = defined_epochs[~bad_epochs_info]
    
    # Separate epochs by attention side condition
    trigger_of_interest = ["left_stim", "right_stim"]
    epochs_sorted = []
        
    for trigger in trigger_of_interest:
        trigger_ids = [ids for ids in defined_epochs.event_id.keys() if ids.startswith(trigger)]
        if trigger_ids:
            epochs_selected = defined_epochs[trigger_ids]
            # inspect epochs 
            avg_trigger_epochs = epochs_selected.copy().average()
            avg_trigger_epochs.plot(picks="MRO51")
            epochs_sorted.append(epochs_selected)
    

    for i, trigger in enumerate(trigger_of_interest):

        cleaned_target_fn = f"sub-{sub}_ses-0{ses}_task-{condition}_meg_{mxf_option}_{trigger}_nrg_ERF_epo.fif"
        save_output_path = os.path.join(no_mxf_path_dir, cleaned_target_fn) 
        epochs_sorted[i].save(save_output_path, overwrite = True)
        print(f"Saved ERF epochs in : {save_output_path}")
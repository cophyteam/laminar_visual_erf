"""
Read in and convert in .fif format raw CTF MEG system files. For memory constrains the recodings are divided in 10min chuncks
"""


import mne
import os 
import glob


def upload_file(rawdata_path, sub, ses, condition):
    """
    Search and upload the raw datachunck

    :param rawdata_path: datapath
    :param sub: subjects id
    :param ses: session id 
    :param condition: experimental condition (block-wise)
    """

    sub_meg_path = os.path.join(rawdata_path, "sub-" + sub  , "ses-0" + ses , "meg/sub-"+ sub + "_ses-0" + ses + "_task-" + condition + "_acq-")
    folders_acq = glob.glob(sub_meg_path + "*_meg.ds", recursive=True)
    list_folders = sorted(folders_acq, key=lambda x: int(x.split('acq-')[1].split('_')[0]))

    return list_folders
    
        
    
if __name__ == "__main__":
    
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

    for s, sub in enumerate(SUB): 
    
        list_folders = upload_file(rawdata_path, sub, ses, condition)
        raws_list = []
        mxf_output_dir = os.path.join(list_folders[0], f"mxf_{mxf_option}_data")
        os.makedirs(mxf_output_dir, exist_ok=True)
        print(f"Subfolder output{mxf_output_dir}")
        
        for acq, acq_file in enumerate(list_folders):
    
            record_num = int(acq_file.split('acq-')[1].split('_')[0])
            print(f"Currently processing sub {sub} ses {ses} condition {condition} recording sess {record_num}")
            raw_read = mne.io.read_raw_ctf(acq_file, preload=True, clean_names=True, verbose=False)
            
            raw = raw_read.copy()
            
            raws_list.append(raw)
        
        # The dev_head_t matrix is different to single session datachunks
        # Take the last session matrix (the final positon of the subject) 
        # Use this matrix for the other datachuncks
        head_info_dest_last = raws_list[-1].info["dev_head_t"]
        
        for file, recording in enumerate(raws_list):
            recording.info["dev_head_t"] = head_info_dest_last
            print(recording.info["dev_head_t"])

        # Concatenate the raw datachunck of the recording session
        raw_concat = mne.concatenate_raws(raws_list, preload=True, on_mismatch='raise', verbose=None)
        print(f"Concatenated {raws_list}")
        
        # save in fif format
        no_mxf_concat_target_fn = f"sub-{sub}_ses-0{ses}_task-{condition}_meg_{mxf_option}.fif"

        no_mxf_save_path = os.path.join(mxf_output_dir, no_mxf_concat_target_fn)
        raw_concat.save(no_mxf_save_path, overwrite = True)

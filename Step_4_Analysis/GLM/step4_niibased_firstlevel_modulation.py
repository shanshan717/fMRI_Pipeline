import os
import pandas as pd
import numpy as np
import os.path as op
import nibabel as nb
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_stat_map
from nilearn.glm.first_level import make_first_level_design_matrix
from itertools import combinations  # For computing contrasts between conditions
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

# Define the function to adjust the event timetable based on motion information
def motion_controlled_event_timetable(event_table, fd_data, six_absolute_motion, TR, FD_thr, ab_motion_thr):
    # Detect timepoints where FD exceeds the threshold
    out_motion_detect = fd_data.to_numpy()[:] > FD_thr
    out_motion_index = np.where(out_motion_detect == True)
    # Detect timepoints where any of the six motion parameters exceed the absolute motion threshold
    six_motion_ex = np.where(np.sum((six_absolute_motion > ab_motion_thr) == True, 1) > 0)
    
    # Convert motion timepoints to actual time by multiplying with TR
    out_motion_time = np.array([])
    if len(out_motion_index[0]) > 0:
        out_motion_time = (out_motion_index[0][:] + 1) * TR
    if len(six_motion_ex[0]) > 0:
        six_motion_time = (six_motion_ex[0] + 1) * TR
        out_motion_time = np.concatenate((out_motion_time, six_motion_time), axis=0)
        out_motion_time = np.unique(out_motion_time)
    
    tmp_timetable = event_table.assign(time_end=lambda dataframe: dataframe['onset'] + dataframe['duration'])
    tmp_timetable = tmp_timetable.reset_index(drop=True)
    
    # Mark the timepoints where motion exceeds thresholds
    block_time_judge = np.zeros(tmp_timetable.shape[0])
    block_time_in = np.zeros(tmp_timetable.shape[0])
    try:
        for n_time in range(tmp_timetable.shape[0]):
            for i in out_motion_time:
                time_judge_0 = (i <= tmp_timetable.loc[n_time, 'time_end'])
                block_time_judge[n_time] += time_judge_0
                time_judge_1 = (i <= tmp_timetable.loc[n_time, 'time_end']) * (i >= tmp_timetable.loc[n_time, 'onset'])
                block_time_in[n_time] += time_judge_1
            
        tmp_timetable = tmp_timetable.assign(
            time_delete=block_time_judge * TR,
            delete_time_inblock=block_time_in
        )
        tmp_timetable.loc[:, 'duration'] = tmp_timetable['duration'] - tmp_timetable['delete_time_inblock'] * TR
        
        # Adjust onset times and recalculate time_end
        for n_time in range(tmp_timetable.shape[0]):
            if n_time != 0:
                tmp_timetable.loc[n_time, 'onset'] = tmp_timetable.loc[n_time, 'onset'] - tmp_timetable.loc[n_time, 'time_delete']
            tmp_timetable.loc[n_time, 'time_end'] = tmp_timetable.loc[n_time, 'onset'] + tmp_timetable.loc[n_time, 'duration']
    except Exception as e:
        print("Error in motion_controlled_event_timetable:", e)
        out_motion_time = False
        tmp_timetable = event_table
    return [tmp_timetable, out_motion_time]

# Define the function to correct NIfTI data based on motion information
def correct_motion_for_niidata(motion_corrected_path, subname, run, task_file, nii_data, TR, out_motion_time):
    motion_corrected_subfolder = op.join(motion_corrected_path, subname)
    if not os.path.exists(motion_corrected_subfolder):
        os.makedirs(motion_corrected_subfolder)
    motion_corrected_nii = op.join(motion_corrected_subfolder, subname + task_file)
    niidata = nb.load(nii_data)
    timepoints_to_delete = ((out_motion_time / TR).astype('int64')) - 1
    motion_corrected_data = np.delete(niidata.get_fdata(), timepoints_to_delete, axis=3)
    motion_corrected_nii_data = nb.Nifti1Image(motion_corrected_data, header=niidata.header, affine=niidata.affine)
    motion_corrected_nii_data.header.set_data_dtype(np.int16)
    nb.save(motion_corrected_nii_data, motion_corrected_nii)
    return motion_corrected_nii

rootdir = '/mnt/d/language_atlas_project/newdata/ds001734'
sublist = ['001']  # Add more subjects as needed
runs = ['01', '02', '03', '04']
taskname = 'MGT'
TR = 1.0  # Adjust TR based on your data
FD_thr = 0.2  # Framewise displacement threshold
ab_motion_thr = 3  # Absolute motion threshold

# Output paths
out_path = op.join(rootdir, 'derivatives', 'first_level_model_corrected_nii')
motion_corrected_path = op.join(rootdir, 'derivatives', 'motion_corrected_data_nii')
# Set whether to perform motion exclusion
do_motion_exclusion = False  # Set to False to avoid deleting frames due to motion
for sub in sublist:
    subname = 'sub-' + sub
    subeventdir = op.join(rootdir, subname, 'func')
    subimagedir = op.join(rootdir, 'derivatives', 'fmriprep', subname, 'func')
    for run in runs:
        sub_event_file = op.join(subeventdir, f'{subname}_task-{taskname}_run-{run}_events.tsv')
        sub_niidata_file = op.join(subimagedir, f'{subname}_task-{taskname}_run-{run}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz')
        sub_motioninfo_file = op.join(subimagedir, f'{subname}_task-{taskname}_run-{run}_bold_confounds.tsv')

        # Check if all necessary files exist
        if not (os.path.exists(sub_event_file) and os.path.exists(sub_niidata_file) and os.path.exists(sub_motioninfo_file)):
            print(f"Missing files for {subname}, run {run}")
            continue

        # Load event data
        event_data = pd.read_csv(sub_event_file, sep='\t')
        # Rename 'participant_response' to 'trial_type', here we use the condition based coding for each trial
        # make trial_type based on whether the participant_response equal to 'NoResp'
        event_data['trial_type'] = event_data['participant_response'].apply(
            lambda x: 'NoResp' if x == 'NoResp' else 'Resp'
        )

        # make sure that gain as modulation effect
        if 'gain' not in event_data.columns:
            print(f"'gain' column not found in {sub_event_file}")
            continue
        else:
            event_data.rename(columns={'gain': 'modulation'}, inplace=True)
        # Load motion parameters data
        confounds = pd.read_csv(sub_motioninfo_file, sep='\t')
        # Handle missing values
        confounds = confounds.fillna(0)

        # Get framewise displacement (FD) data
        if 'FramewiseDisplacement' in confounds.columns:
            fd = confounds[['FramewiseDisplacement']]
        elif 'framewise_displacement' in confounds.columns:
            fd = confounds[['framewise_displacement']]
        else:
            print(f"FD column not found in {sub_motioninfo_file}")
            continue

        # Get six motion parameters
        motion_params = confounds[['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']]

        # Motion correction
        if do_motion_exclusion:
            [event_data_corrected, out_motion_time] = motion_controlled_event_timetable(event_data, fd, motion_params, TR, FD_thr, ab_motion_thr)
            # Save corrected event data
            out_sub_path = op.join(out_path, subname)
            if not os.path.exists(out_sub_path):
                os.makedirs(out_sub_path)
            event_out_file = op.join(out_sub_path, f'{subname}_task-{taskname}_run-{run}_events_corrected.tsv')
            event_data_corrected.to_csv(event_out_file, sep='\t', index=False)
            # Correct NIfTI data
            if not isinstance(out_motion_time, bool):
                task_file = f'_task-{taskname}_run-{run}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'
                corrected_nii_file = correct_motion_for_niidata(motion_corrected_path, subname, run, task_file, sub_niidata_file, TR, out_motion_time)
                fmri_img = nb.load(corrected_nii_file)
                # Adjust motion parameters
                timepoints_to_delete = ((out_motion_time / TR).astype('int64')) - 1
                motion_params_corrected = motion_params.drop(motion_params.index[timepoints_to_delete]).reset_index(drop=True)
            else:
                fmri_img = nb.load(sub_niidata_file)
                motion_params_corrected = motion_params
                event_data_corrected = event_data
        else:
            event_data_corrected = event_data
            fmri_img = nb.load(sub_niidata_file)
            motion_params_corrected = motion_params
            
        frame_times = (
            np.arange(fmri_img.shape[3]) * TR
        )  # here are the corresponding frame times
        
        unmodulated_matrix = event_data_corrected[['onset', 'duration', 'trial_type']]
        modulated_matrix = event_data_corrected[['onset', 'duration', 'trial_type','modulation']]
        
        GLM_matrix_unmodulated = make_first_level_design_matrix(
            frame_times,
            unmodulated_matrix,
            drift_model="polynomial",
            drift_order=3,
            hrf_model="spm + derivative",
        )
        
        GLM_matrix_modulated = make_first_level_design_matrix(
            frame_times,
            modulated_matrix,
            drift_model="polynomial",
            drift_order=3,
            hrf_model="spm + derivative",
        )
        # Let's compare two design matrix
        fig, (ax1, ax2) = plt.subplots(
            figsize=(10, 6), nrows=1, ncols=2, constrained_layout=True
        )

        plot_design_matrix(GLM_matrix_unmodulated, axes=ax1,rescale=False)
        ax1.set_title("Event design matrix", fontsize=12)
        plot_design_matrix(GLM_matrix_modulated, axes=ax2,rescale=False)
        ax2.set_title("Modulated Event design matrix", fontsize=12)
        plt.show()

        # Perform first-level GLM analysis
        fmri_glm = FirstLevelModel(
            t_r=TR,
            noise_model='ar1',
            hrf_model='spm + derivative',
            drift_model='polynomial',# the desired drift model for the design matrices
            drift_order = 3,  # Adjust the drift order as needed
            high_pass=1./128,  # Adjust the high-pass filter as needed
            signal_scaling=False,  # Whether to scale the signal
            minimize_memory=False
        )
        # we use the GLM_matrix_modulated
        fmri_glm = fmri_glm.fit(
            fmri_img,
            events=modulated_matrix,
            confounds=motion_params_corrected
        )

        # Define contrasts
        # Assuming 'trial_type' column contains condition names
        conditions = event_data_corrected['trial_type'].unique()
        design_matrix = fmri_glm.design_matrices_[0]
        # Create contrasts for each condition
        contrasts = {}
        for cond in conditions:
            # The columns corresponding to the condition
            cond_vector = np.array([1 if c == cond else 0 for c in design_matrix.columns])
            contrasts[cond] = cond_vector
        # Compute contrasts for each condition vs baseline
        out_sub_path = op.join(out_path, subname)
        stats_results_path = op.join(out_sub_path, 'stats_results', f'run-{run}')
        if not os.path.exists(stats_results_path):
            os.makedirs(stats_results_path)
        for contrast_id, contrast_val in contrasts.items():
            z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score')
            z_map_file = op.join(stats_results_path, f'{subname}_task-{taskname}_run-{run}_{contrast_id}_zmap.nii.gz')
            z_map.to_filename(z_map_file)
            print(f"Contrast {contrast_id} for {subname}, run {run} completed and saved to {z_map_file}")
            # Optionally, plot the contrast map
            plot_stat_map(z_map, title=f'{subname} {contrast_id}', display_mode='ortho', threshold=1.0)# unthreshold stats map
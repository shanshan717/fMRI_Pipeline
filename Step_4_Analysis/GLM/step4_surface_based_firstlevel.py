import os
import pandas as pd
import numpy as np
import os.path as op
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import nibabel as nb
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import compute_contrast
from itertools import combinations  # For computing contrasts between conditions

# Define the function to adjust the event timetable based on motion information
def motion_controlled_event_timetable(event_table, fd_data, six_absolute_motion, TR, FD_thr, ab_motion_thr):
    # Detect timepoints where FD exceeds the threshold
    out_motion_detect = fd_data.to_numpy().flatten() > FD_thr
    out_motion_index = np.where(out_motion_detect)[0]
    # Detect timepoints where any of the six motion parameters exceed the absolute motion threshold
    six_motion_ex = np.where(np.any(np.abs(six_absolute_motion) > ab_motion_thr, axis=1))[0]
    
    # Convert motion timepoints to actual time by multiplying with TR
    out_motion_time = np.array([])
    if len(out_motion_index) > 0:
        out_motion_time = (out_motion_index + 1) * TR
    if len(six_motion_ex) > 0:
        six_motion_time = (six_motion_ex + 1) * TR
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
                time_judge_1 = (i <= tmp_timetable.loc[n_time, 'time_end']) and (i >= tmp_timetable.loc[n_time, 'onset'])
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

# Define the function to correct motion in GIFTI data
def correct_motion_for_giidata(motion_corrected_path, subname, run, task_file_L, task_file_R, data_L_path, data_R_path, TR, out_motion_time):
    motion_corrected_subfolder = op.join(motion_corrected_path, subname)
    if not os.path.exists(motion_corrected_subfolder):
        os.makedirs(motion_corrected_subfolder)
    corrected_gii_file_L = op.join(motion_corrected_subfolder, subname + task_file_L)
    corrected_gii_file_R = op.join(motion_corrected_subfolder, subname + task_file_R)
    
    # Load GIFTI data
    data_L = nb.load(data_L_path)
    data_R = nb.load(data_R_path)
    
    # Calculate timepoints to delete
    timepoints_to_delete = ((out_motion_time / TR).astype(int)) - 1
    timepoints_to_keep = np.setdiff1d(np.arange(len(data_L.darrays)), timepoints_to_delete)
    
    # Create new GIFTI images with selected timepoints
    corrected_darrays_L = [data_L.darrays[i] for i in timepoints_to_keep]
    corrected_darrays_R = [data_R.darrays[i] for i in timepoints_to_keep]
    
    corrected_data_L = nb.gifti.GiftiImage(darrays=corrected_darrays_L)
    corrected_data_R = nb.gifti.GiftiImage(darrays=corrected_darrays_R)
    
    # Save corrected GIFTI data
    nb.save(corrected_data_L, corrected_gii_file_L)
    nb.save(corrected_data_R, corrected_gii_file_R)
    
    return corrected_gii_file_L, corrected_gii_file_R

# Main code
rootdir = '/mnt/d/language_atlas_project/newdata/ds001734'
sublist = ['001']  # Add more subjects as needed
runs = ['01', '02', '03', '04']
taskname = 'MGT'
TR = 1.0  # Adjust TR based on your data
FD_thr = 0.2  # Framewise displacement threshold
ab_motion_thr = 3  # Absolute motion threshold

# Output paths
out_path = op.join(rootdir, 'derivatives', 'first_level_model_corrected_gii')
motion_corrected_path = op.join(rootdir, 'derivatives', 'motion_corrected_data_gii')

# Set whether to perform motion exclusion
do_motion_exclusion = False  # Set to False to avoid deleting frames due to motion

for sub in sublist:
    subname = 'sub-' + sub
    subeventdir = op.join(rootdir, subname, 'func')
    subimagedir = op.join(rootdir, 'derivatives', 'fmriprep', subname, 'func')
    for run in runs:
        sub_event_file = op.join(subeventdir, f'{subname}_task-{taskname}_run-{run}_events.tsv')
        sub_gii_file_L = op.join(subimagedir, f'{subname}_task-{taskname}_run-{run}_bold_space-fsaverage5.L.func.gii')
        sub_gii_file_R = op.join(subimagedir, f'{subname}_task-{taskname}_run-{run}_bold_space-fsaverage5.R.func.gii')
        sub_motioninfo_file = op.join(subimagedir, f'{subname}_task-{taskname}_run-{run}_bold_confounds.tsv')

        # Check if all necessary files exist
        if not (os.path.exists(sub_event_file) and os.path.exists(sub_gii_file_L) and os.path.exists(sub_gii_file_R) and os.path.exists(sub_motioninfo_file)):
            print(f"Missing files for {subname}, run {run}")
            continue

        # Load event data
        event_data = pd.read_csv(sub_event_file, sep='\t')
        # Rename 'participant_response' to 'trial_type'
        if 'participant_response' in event_data.columns:
            event_data.rename(columns={'participant_response': 'trial_type'}, inplace=True)
        else:
            print(f"'participant_response' column not found in {sub_event_file}")
            continue

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
            # [Motion correction code remains unchanged]
            # ...
            pass
        else:
            event_data_corrected = event_data
            data_L = nb.load(sub_gii_file_L)
            data_R = nb.load(sub_gii_file_R)
            motion_params_corrected = motion_params

        # Concatenate left and right hemisphere data
        n_timepoints = len(data_L.darrays)
        n_vertices_L = data_L.darrays[0].data.shape[0]
        n_vertices_R = data_R.darrays[0].data.shape[0]

        # Initialize data matrix
        data_matrix = np.zeros((n_timepoints, n_vertices_L + n_vertices_R))
        for t in range(n_timepoints):
            data_L_t = data_L.darrays[t].data
            data_R_t = data_R.darrays[t].data
            data_matrix[t, :n_vertices_L] = data_L_t
            data_matrix[t, n_vertices_L:] = data_R_t

        # Build design matrix
        frame_times = TR * (np.arange(n_timepoints))
        design_matrix = make_first_level_design_matrix(
            frame_times,
            event_data_corrected,
            drift_model='polynomial',
            drift_order=3,
            add_regs=motion_params_corrected,
            add_reg_names=motion_params_corrected.columns,
            hrf_model='spm'
        )

        # Perform GLM analysis
        # Do NOT transpose data_matrix; Y should have shape (n_samples, n_voxels)
        labels, estimates = run_glm(data_matrix, design_matrix.values)

        # Define contrasts
        conditions = event_data_corrected['trial_type'].unique()
        design_columns = design_matrix.columns

        # Create contrasts for each condition
        contrasts = {}
        for cond in conditions:
            # The columns corresponding to the condition
            cond_vector = np.array([1 if cond == col else 0 for col in design_columns])
            contrasts[cond] = cond_vector

        # Prepare output directories
        out_sub_path = op.join(out_path, subname)
        stats_results_path = op.join(out_sub_path, 'stats_results', f'run-{run}')
        if not os.path.exists(stats_results_path):
            os.makedirs(stats_results_path)

        # Compute contrasts for each condition vs baseline
        for contrast_id, contrast_val in contrasts.items():
            contrast = compute_contrast(labels, estimates, contrast_val)
            # Compute Z-map
            z_map = contrast.z_score()
            # z_map has shape (n_voxels,)
            # Split z_map back to left and right hemispheres
            z_map_L = z_map[:n_vertices_L]
            z_map_R = z_map[n_vertices_L:]

            # Create GIFTI DataArrays
            z_map_L_darray = nb.gifti.GiftiDataArray(data=np.int32(z_map_L))
            z_map_R_darray = nb.gifti.GiftiDataArray(data=np.int32(z_map_R))

            # Create GIFTI images
            z_map_img_L = nb.gifti.GiftiImage(darrays=[z_map_L_darray])
            z_map_img_R = nb.gifti.GiftiImage(darrays=[z_map_R_darray])

            # Save GIFTI images
            z_map_file_L = op.join(stats_results_path, f'{subname}_task-{taskname}_run-{run}_{contrast_id}_zmap.L.func.gii')
            z_map_file_R = op.join(stats_results_path, f'{subname}_task-{taskname}_run-{run}_{contrast_id}_zmap.R.func.gii')
            nb.save(z_map_img_L, z_map_file_L)
            nb.save(z_map_img_R, z_map_file_R)

            print(f"Contrast {contrast_id} for {subname}, run {run} completed and saved to {z_map_file_L} and {z_map_file_R}")
import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo):
    taskrun1 = create_key('sub-{subject}/ses-1/func/sub-{subject}_ses-1_task-wordpic_run1_bold')
    t1w = create_key('sub-{subject}/ses-1/anat/sub-{subject}_ses-1_run1_T1w')
    dwi = create_key('sub-{subject}/ses-1/dwi/sub-{subject}_ses-1_run1_dwi')
    
    info = {taskrun1: [], t1w: [], dwi: []}
    
    for s in seqinfo:
        if ('3_AxBOLD-1' in s.dcm_dir_name):
            info[taskrun1] = [s.series_id]
        elif ('5_Ax3DT1BRAVO' in s.dcm_dir_name):
            info[t1w] = [s.series_id]
        elif ('7_AxDTI64Directions' in s.dcm_dir_name):
            info[dwi] = [s.series_id]
    return info
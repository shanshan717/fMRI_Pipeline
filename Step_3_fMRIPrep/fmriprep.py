import os

subs = ['014', '015', '017']  # 被试列表
bids_root_dir = "/mnt/sdb1/Judge-fMRI-Data/Bids"  # BIDS 数据根目录
bids_out_dir = "/mnt/sdb1/Judge-fMRI/fmriprep"    # 输出目录
bids_work_dir = "/mnt/sdb1/Judge-fMRI/fmriprep_wd" # 工作目录
license_path = "/mnt/sdb1/Judge-fMRI/Scripts/ProcessingScripts/fMRIPrep/license.txt"
nthreads = 12
mem_mb = 32000

for sub in subs:
    output_dir = os.path.join(bids_out_dir, f"sub-{sub}")
    work_dir = os.path.join(bids_work_dir, f"sub-{sub}")

    # 创建输出目录和工作目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    cmd = f"""
    docker run --rm -it \
        -v {bids_root_dir}:/inputbids \
        -v {output_dir}:/output \
        -v {work_dir}:/wd \
        -v {os.path.dirname(license_path)}:/freesurfer_license \
        nipreps/fmriprep:23.0.2 \
        /inputbids /output participant \
        --participant_label {sub} \
        -w /wd \
        --nthreads {nthreads} --omp-nthreads {nthreads} \
        --mem-mb {mem_mb} \
        --fs-license-file /freesurfer_license/{os.path.basename(license_path)} \
        --output-spaces T1w MNI152NLin6Asym MNI152NLin2009cAsym \
        --ignore slicetiming \
        --return-all-components \
        --notrack \
        --skip-bids-validation \
        --debug all \
        --stop-on-first-crash \
        --resource-monitor \
        --cifti-output 91k \
        --verbose
    """
    print(f"Running fMRIPrep for subject {sub}...")
    os.system(cmd)

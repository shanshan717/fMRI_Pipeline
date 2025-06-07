import os
subs = ['014', '015', '017']
bids_root_dir = "/mnt/sdb1/Judge-fMRI-Data/Bids"
bids_out_dir = "/mnt/sdb1/Judge-fMRI/fmriprep"
license_path = "/mnt/sdb1/Judge-fMRI/Scripts/ProcessingScripts/fMRIPrep/license.txt"
nthreads = 12

print("Checking fMRIPrep version...")
os.system("docker run -rm -it nipreps/fmriprep --version")

for sub in subs:
    cmd = f"""
    docker run --rm -it \
        -v {bids_root_dir}:/inputbids \
        -v {bids_out_dir}/fmriprep/sub-{sub}:/output \
        -v {license_path}:/license.txt \
        nipreps/fmriprep \
        /inputbids /output participant \
        --participant_label {sub} \
        --nthreads {nthreads} --omp-nthreads {nthreads} \
        --mem-mb 32000 --ignore slicetiming --ignore fieldmaps \
        --fs-license-file /license.txt \
        --verbose
    """
    print(f"Running fMRIPrep for subject {sub}")
    os.system(cmd)

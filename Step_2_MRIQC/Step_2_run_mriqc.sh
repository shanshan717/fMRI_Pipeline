#!/bin/bash

# User inputs:
bids_root_dir=/Volumes/ss/fMRI_Pipeline/BIDS
subj=$1
nthreads=6
mem=24  # memory in GB

# Define output path
output_dir=$bids_root_dir/derivatives/mriqc

# Run MRIQC
echo ""
echo "Running MRIQC on participant: sub-$subj"
echo ""

# Make output directories if they don't exist
mkdir -p $output_dir/sub-${subj}

# Run MRIQC with the provided Docker image and settings
docker run -it --rm \
  -v $bids_root_dir:/data:ro \
  -v $output_dir/sub-${subj}:/out \
  poldracklab/mriqc:latest /data /out participant \
  --participant_label $subj \
  --n_proc $nthreads \
  --mem_gb $mem \
  --float32 \
  --ants-nthreads $nthreads \
  -w /out \
  --verbose-reports \
  --no-sub


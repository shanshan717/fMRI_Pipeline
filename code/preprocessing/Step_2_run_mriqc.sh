#!/bin/bash
bids_root_dir=/mnt/sdb1/Judge-fMRI-Data/bids
output_dir=/mnt/sdb1/Judge-fMRI-Data/MRIQC_SS
nthreads=6
mem=24 # memory in GB

# List of participant IDs
participants=("036" "037" "038" "039" "040" "043" "044" "045" 
"046" "047" "049" "050" "051" "052" "053" "054" "055")

total=${#participants[@]}  
count=0  

# Progress bar function
progress_bar() {
  local progress=$1
  local total=$2
  local width=50  
  local percent=$(( progress * 100 / total ))
  local completed=$(( progress * width / total ))
  local remaining=$(( width - completed ))

  printf "["
  for ((i = 0; i < completed; i++)); do
    printf "="
  done
  for ((i = 0; i < remaining; i++)); do
    printf " "
  done
  printf "] %d%% (%d/%d)\r" $percent $progress $total
}

# Loop through each participant
for subj in "${participants[@]}"
do
  count=$((count + 1))  

  echo ""
  echo "Running MRIQC on participant: sub-$subj"
  echo ""

  # Create the MRIQC derivatives folder and participant-specific folder if they don't exist
  if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
  fi

  if [ ! -d $output_dir/sub-${subj} ]; then
    mkdir -p $output_dir/sub-${subj}
  fi

  # Run MRIQC with the provided Docker image and settings
  docker run -it --rm \
    -v $bids_root_dir:/data:ro \
    -v $output_dir/sub-${subj}:/out \
  ss/mriqc /data /out participant \
    --participant_label $subj \
    --n_proc $nthreads \
    --mem_gb $mem \
    --float32 \
    --ants-nthreads $nthreads \
    -w /out \
    --verbose-reports \
    --no-sub

  progress_bar $count $total
  echo ""  
done

echo "All participants processed."

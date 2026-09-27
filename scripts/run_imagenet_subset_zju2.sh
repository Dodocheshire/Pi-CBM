#!/usr/bin/env bash
# Run on ZJU-GPU2. The downloader resumes from the last completed shard.
set -u

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

root=/home/pengcheng/VLG-CBM/datasets/imagenet/ILSVRC/Data/CLS-LOC
python=/home/pengcheng/miniconda3/envs/pi-cbm/bin/python
script=/home/pengcheng/Pi-CBM/scripts/download_imagenet_subset.py
output="$root/ImageNet_train_200_10_clean"
proxy_args=()
if [[ -n "${PI_CBM_SOCKS_PROXY:-}" ]]; then
    proxy_args=(--proxy "$PI_CBM_SOCKS_PROXY")
fi

for attempt in $(seq 1 30); do
    echo "$(date -Is) attempt=$attempt"
    if "$python" -u "$script" \
        --output "$output" \
        --scratch /dev/shm/pi-cbm-imagenet-clean \
        --reference-val "$root/ImageNet_val" \
        "${proxy_args[@]}"; then
        echo "$(date -Is) complete"
        exit 0
    fi
    echo "$(date -Is) download interrupted; resuming in 30 seconds"
    sleep 30
done

echo "$(date -Is) failed after 30 attempts"
exit 1

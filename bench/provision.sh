#!/usr/bin/env bash
# Provision the FreeToken disk-tier bench VM: 1xL4 (24GB, Ada sm_89), 128GB RAM,
# 375GB local NVMe. Spot, STOP on preemption so a long bench can resume.
set -euo pipefail

PROJECT="${PROJECT:-h0melab}"
ZONE="${ZONE:-europe-west2-a}"
NAME="${NAME:-freetoken-bench}"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type=g2-standard-32 \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --local-ssd=interface=NVME \
  --network=homelab-hub --subnet=homelab-hub-lon

echo "SSH:    gcloud compute ssh $NAME --zone=$ZONE"
echo "Setup:  copy vm-setup.sh over and run it"
echo "DELETE WHEN DONE: gcloud compute instances delete $NAME --zone=$ZONE"

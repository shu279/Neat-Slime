# Run Training On Google Compute Engine

This project runs as a CPU job on your Debian 12 `e2-standard-8` VM.

## Copy Project To VM

```bash
gcloud compute ssh gce-shu0 \
  --project gcp-samples-ic0 \
  --zone asia-northeast1-c \
  --command "mkdir -p ~/neat-slime"

gcloud compute scp --recurse \
  src slimevolleygym saved scripts requirements-gce.txt GCE_RUN.md \
  gce-shu0:~/neat-slime \
  --project gcp-samples-ic0 \
  --zone asia-northeast1-c
```

## Setup VM

```bash
gcloud compute ssh gce-shu0 \
  --project gcp-samples-ic0 \
  --zone asia-northeast1-c

cd ~/neat-slime
bash scripts/setup_gce.sh
```

## Train

```bash
cd ~/neat-slime
bash scripts/train_gce.sh
```

For this `e2-standard-8` VM, use `NUM_WORKERS=4` by default. The VM exposes
8 vCPUs, but they are 4 physical cores with hyperthreading, so using more than
4 Python workers can be slower.

Useful shorter test run:

```bash
GENERATIONS=50 NUM_WORKERS=4 bash scripts/train_gce.sh
```

One-command local roundtrip:

```bash
GENERATIONS=50 NUM_WORKERS=4 bash scripts/gce_roundtrip.sh
```

Omit `GENERATIONS=50` for the full default 500-generation run.

The roundtrip script starts training with `nohup` on the VM and then polls for
completion. If the local SSH polling connection drops, training should continue
on the VM. Re-run the script only after checking that no training process is
already active.

## Keep Training Running After Disconnect

```bash
cd ~/neat-slime
nohup bash scripts/train_gce.sh > logs/nohup-train.log 2>&1 &
tail -f logs/nohup-train.log
```

## Copy Results Back

```bash
gcloud compute scp --recurse \
  gce-shu0:~/neat-slime/saved \
  ./saved-from-gce \
  --project gcp-samples-ic0 \
  --zone asia-northeast1-c

gcloud compute scp --recurse \
  gce-shu0:~/neat-slime/logs \
  ./logs-from-gce \
  --project gcp-samples-ic0 \
  --zone asia-northeast1-c
```

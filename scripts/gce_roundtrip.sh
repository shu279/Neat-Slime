#!/usr/bin/env bash
set -euo pipefail

project="${GCE_PROJECT:-gcp-samples-ic0}"
zone="${GCE_ZONE:-asia-northeast1-c}"
instance="${GCE_INSTANCE:-gce-shu0}"
remote_dir="${GCE_REMOTE_DIR:-/home/shusato/neat-slime}"

timestamp="$(date +%Y%m%d-%H%M%S)"
stage_dir="gce-results-${timestamp}"
upload_dir="$(mktemp -d "${TMPDIR:-/tmp}/neat-slime-upload.XXXXXX")"

cleanup() {
  rm -rf "${upload_dir}"
}
trap cleanup EXIT

echo "Uploading current code to ${instance}:${remote_dir}"
mkdir -p "${upload_dir}/slimevolleygym"
cp -R src scripts "${upload_dir}/"
cp slimevolleygym/setup.py "${upload_dir}/slimevolleygym/"
cp -R slimevolleygym/slimevolleygym "${upload_dir}/slimevolleygym/"
cp requirements-gce.txt GCE_RUN.md "${upload_dir}/"
find "${upload_dir}" -name .git -type d -prune -exec rm -rf {} +
find "${upload_dir}" -name __pycache__ -type d -prune -exec rm -rf {} +
find "${upload_dir}" -name "*.pyc" -type f -delete
find "${upload_dir}" -name "*.mp4" -type f -delete
find "${upload_dir}" -name "*.gif" -type f -delete

gcloud compute ssh "${instance}" \
  --project "${project}" \
  --zone "${zone}" \
  --command "mkdir -p ${remote_dir}"

gcloud compute scp --recurse \
  "${upload_dir}/src" \
  "${upload_dir}/slimevolleygym" \
  "${upload_dir}/scripts" \
  "${upload_dir}/requirements-gce.txt" \
  "${upload_dir}/GCE_RUN.md" \
  "${instance}:${remote_dir}" \
  --project "${project}" \
  --zone "${zone}"

echo "Running training on ${instance}"
gcloud compute ssh "${instance}" \
  --project "${project}" \
  --zone "${zone}" \
  --command "cd ${remote_dir} && test -x .venv/bin/python || { echo 'ERROR: ${remote_dir}/.venv is missing or incomplete. Run VM setup first.' >&2; exit 2; } && bash scripts/train_gce.sh && PYTHONPATH=.:./slimevolleygym .venv/bin/python -m src.evaluate_saved --path saved/best_genome.json --record-mp4 saved/best_gameplay.mp4"

echo "Downloading VM results to ${stage_dir}"
mkdir -p "${stage_dir}"
gcloud compute scp --recurse \
  "${instance}:${remote_dir}/saved" \
  "${stage_dir}" \
  --project "${project}" \
  --zone "${zone}"

echo "Replacing local saved with VM results"
rm -rf saved
if [ -d "${stage_dir}/saved" ]; then
  mv "${stage_dir}/saved" saved
  rmdir "${stage_dir}"
else
  mv "${stage_dir}" saved
fi

echo "Done. Local saved now contains the VM results."

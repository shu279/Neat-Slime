#!/usr/bin/env bash
set -euo pipefail

project="${GCE_PROJECT:-gcp-samples-ic0}"
zone="${GCE_ZONE:-asia-northeast1-c}"
instance="${GCE_INSTANCE:-gce-shu0}"
remote_dir="${GCE_REMOTE_DIR:-/home/shusato/neat-slime}"

timestamp="$(date +%Y%m%d-%H%M%S)"
stage_dir="gce-results-${timestamp}"
upload_dir="$(mktemp -d "${TMPDIR:-/tmp}/neat-slime-upload.XXXXXX")"
remote_status_file="logs/roundtrip-${timestamp}.status"
remote_pid_file="logs/roundtrip-${timestamp}.pid"
remote_driver_log="logs/roundtrip-${timestamp}.log"

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

echo "Starting detached training on ${instance}"
gcloud compute ssh "${instance}" \
  --project "${project}" \
  --zone "${zone}" \
  --command "cd ${remote_dir} && test -x .venv/bin/python || { echo 'ERROR: ${remote_dir}/.venv is missing or incomplete. Run VM setup first.' >&2; exit 2; } && mkdir -p logs && rm -f ${remote_status_file} ${remote_pid_file} ${remote_driver_log} && { nohup bash -c 'bash scripts/train_gce.sh; code=\$?; echo \${code} > ${remote_status_file}; exit \${code}' > ${remote_driver_log} 2>&1 < /dev/null & echo \$! > ${remote_pid_file}; echo \"Started detached training pid \$(cat ${remote_pid_file})\"; }"

echo "Polling detached training status. SSH polling interruptions will not stop training."
last_log_line=0
while true; do
  set +e
  poll_output="$(
    gcloud compute ssh "${instance}" \
      --project "${project}" \
      --zone "${zone}" \
      --command "cd ${remote_dir} && if test -f ${remote_status_file}; then echo STATUS:\$(cat ${remote_status_file}); else latest=\$(ls -t logs/train-*.log 2>/dev/null | head -n 1); if test -n \"\${latest}\"; then total=\$(wc -l < \"\${latest}\"); echo RUNNING lines:\${total}; if test \${total} -gt ${last_log_line}; then tail -n +$((last_log_line + 1)) \"\${latest}\"; fi; else echo RUNNING; fi; fi" 2>&1
  )"
  poll_code=$?
  set -e

  echo "${poll_output}"
  new_log_line="$(printf "%s\n" "${poll_output}" | sed -n 's/^RUNNING lines://p' | tail -n 1)"
  if [ -n "${new_log_line}" ]; then
    last_log_line="${new_log_line}"
  fi

  if [ "${poll_code}" -ne 0 ]; then
    echo "Polling SSH failed; training remains detached on the VM. Retrying in 30 seconds."
    sleep 30
    continue
  fi

  if printf "%s\n" "${poll_output}" | grep -q "^STATUS:"; then
    remote_status="$(printf "%s\n" "${poll_output}" | sed -n 's/^STATUS://p' | tail -n 1)"
    if [ "${remote_status}" = "0" ]; then
      echo "Detached training finished successfully."
      break
    fi
    echo "Detached training failed with exit code ${remote_status}."
    echo "Remote driver log: ${remote_dir}/${remote_driver_log}"
    exit "${remote_status}"
  fi

  sleep 30
done

echo "Generating gameplay video on ${instance}"
gcloud compute ssh "${instance}" \
  --project "${project}" \
  --zone "${zone}" \
  --command "cd ${remote_dir} && PYTHONPATH=.:./slimevolleygym .venv/bin/python -m src.evaluate_saved --path saved/best_genome.json --record-mp4 saved/best_gameplay.mp4"

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

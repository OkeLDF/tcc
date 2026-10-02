#!/usr/bin/env bash
# Runs the downstream fine-tuning for each label schema and uploads the
# results of each run to a private Hugging Face model repo:
#   bethesda      -> okeldf/vit-cytology-bethesda
#   morphological -> okeldf/vit-cytology-morphological
#
# Uploaded per run, under <FROM_PRETRAINED>/ (e.g. local/ or base/):
#   weights/  best checkpoint of each phase and the last checkpoint
#   *.csv     training history and test results
#   logs/     this run's part of training.log and the full console output
#   configs.yaml
#
# Usage (from the repo root, with the conda env active):
#   ./run_finetuning.sh
# Requires `hf auth login` with a token that can write to the okeldf namespace.

set -euo pipefail

HF_NAMESPACE="okeldf"
SCHEMAS=(bethesda morphological)
PYTHON="${PYTHON:-python}"
UPLOAD_RETRIES=3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"

repo_id() { echo "$HF_NAMESPACE/vit-cytology-$1"; }

config_value() {  # config_value KEY [SUBKEY]
    "$PYTHON" -c "import sys, yaml
value = yaml.safe_load(open('$SRC/configs.yaml'))
for key in sys.argv[1:]:
    value = value[key]
print(value)" "$@"
}

upload_run() {  # upload_run SCHEMA CONDITION STAGING_DIR
    local repo; repo="$(repo_id "$1")"
    for attempt in $(seq 1 "$UPLOAD_RETRIES"); do
        if hf upload "$repo" "$3" "$2" --repo-type model \
            --commit-message "Fine-tuning $1 ($2) - $(date '+%Y-%m-%d %H:%M')"; then
            return 0
        fi
        echo "Upload to $repo failed (attempt $attempt/$UPLOAD_RETRIES)." >&2
        sleep 30
    done
    return 1
}

# Check the login and create the repos before training, so that a permission
# problem shows up now and not after hours of training.
echo "Hugging Face login: $(hf auth whoami)"
for schema in "${SCHEMAS[@]}"; do
    hf repos create "$(repo_id "$schema")" --private --exist-ok
done

condition="$(config_value DOWNSTREAM FROM_PRETRAINED)"
log_dir="$ROOT/$(config_value PROJECT_LOG)"
finetuned_dir="$ROOT/$(config_value PATH_FINETUNED)"
mkdir -p "$log_dir"
failed_uploads=()

cd "$SRC"  # log.py writes training.log to the working directory
for schema in "${SCHEMAS[@]}"; do
    stamp="$(date '+%Y%m%d_%H%M%S')"
    console_log="$log_dir/console_${schema}_${condition}_${stamp}.log"
    log_offset=$(stat -c %s training.log 2>/dev/null || echo 0)

    echo "=== Fine-tuning: schema=$schema, FROM_PRETRAINED=$condition ==="
    "$PYTHON" run.py downstream --schema "$schema" 2>&1 | tee "$console_log"

    staging="$(mktemp -d "$ROOT/.upload_${schema}_XXXX")"
    mkdir -p "$staging/weights" "$staging/logs"
    cp -r "$finetuned_dir/$schema/$condition/." "$staging/weights/"
    cp "$log_dir/downstream_history_${schema}_${condition}.csv" "$staging/"
    cp "$log_dir/downstream_test_${schema}_${condition}.csv" "$staging/"
    tail -c +"$((log_offset + 1))" training.log > "$staging/logs/training.log"
    cp "$console_log" "$staging/logs/"
    cp configs.yaml "$staging/"

    if upload_run "$schema" "$condition" "$staging"; then
        rm -rf "$staging"
    else
        echo "Upload of $schema failed; files kept in $staging" >&2
        failed_uploads+=("$schema")
    fi
done

if ((${#failed_uploads[@]})); then
    echo "Uploads that failed: ${failed_uploads[*]}" >&2
    exit 1
fi
echo "Done."

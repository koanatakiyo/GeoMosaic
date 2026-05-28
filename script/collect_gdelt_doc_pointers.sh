#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PYTHON_BIN:-python}"
events_csv="hongkong,ukraine"
max_records=50
window_days=14
sleep_seconds=6
gdelt_delay_seconds=5
gdelt_retries=4
gdelt_retry_backoff_seconds=15
output_dir="data/0_external/external_asset_raw"
dry_run=0
merge_after=0
failed_events=()

usage() {
  cat <<'EOF'
Usage: script/collect_gdelt_doc_pointers.sh [options]

Sequentially collect GDELT DOC article pointers with conservative spacing.

Default events are hongkong,ukraine because GDELT DOC supports event-window
coverage for these recent events. Use --all-events to also collect
retrospective pointers for older events.

Options:
  --events CSV                       Event ids to collect.
  --all-events                       Collect all registered Tier-1 events.
  --max-records N                    GDELT DOC maxrecords per event. Default: 50
  --window-days N                    Event-window radius for recent events. Default: 14
  --sleep-seconds N                  Sleep between event commands. Default: 6
  --gdelt-delay-seconds N            In-process GDELT delay. Default: 5
  --gdelt-retries N                  Retries for HTTP 429. Default: 4
  --gdelt-retry-backoff-seconds N    Initial retry backoff for HTTP 429. Default: 15
  --output-dir DIR                   Raw output directory. Default: data/0_external/external_asset_raw
  --merge-after                      Rebuild data/0_external/external_assets.jsonl after collection.
  --dry-run                          Print commands without executing them.
  -h, --help                         Show this help.
EOF
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$dry_run" == "1" ]]; then
    return 0
  fi
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --events)
      events_csv="$2"
      shift 2
      ;;
    --all-events)
      events_csv="crimea,iraq,libya,kosovo,scs,jcpoa,hongkong,ukraine"
      shift
      ;;
    --max-records)
      max_records="$2"
      shift 2
      ;;
    --window-days)
      window_days="$2"
      shift 2
      ;;
    --sleep-seconds)
      sleep_seconds="$2"
      shift 2
      ;;
    --gdelt-delay-seconds)
      gdelt_delay_seconds="$2"
      shift 2
      ;;
    --gdelt-retries)
      gdelt_retries="$2"
      shift 2
      ;;
    --gdelt-retry-backoff-seconds)
      gdelt_retry_backoff_seconds="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --merge-after)
      merge_after=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$output_dir"
IFS=',' read -r -a events <<< "$events_csv"

for index in "${!events[@]}"; do
  event="${events[$index]}"
  output_path="${output_dir%/}/gdelt_doc_${event}.jsonl"
  if ! run_cmd "$python_bin" script/fetch_gdelt_doc_assets.py \
    --event "$event" \
    --max-records "$max_records" \
    --window-days "$window_days" \
    --gdelt-delay-seconds "$gdelt_delay_seconds" \
    --gdelt-retries "$gdelt_retries" \
    --gdelt-retry-backoff-seconds "$gdelt_retry_backoff_seconds" \
    --output "$output_path"; then
    echo "warning: GDELT DOC collection failed for event=${event}; continuing." >&2
    failed_events+=("$event")
  fi

  if [[ "$index" -lt "$((${#events[@]} - 1))" && "$sleep_seconds" != "0" && "$sleep_seconds" != "0.0" ]]; then
    run_cmd sleep "$sleep_seconds"
  fi
done

if [[ "$merge_after" == "1" ]]; then
  run_cmd "$python_bin" script/collect_external_assets.py \
    --collect-existing "$output_dir"
fi

if [[ "${#failed_events[@]}" -gt 0 ]]; then
  joined="$(IFS=','; echo "${failed_events[*]}")"
  echo "GDELT DOC collection failed for events: ${joined}" >&2
  exit 1
fi

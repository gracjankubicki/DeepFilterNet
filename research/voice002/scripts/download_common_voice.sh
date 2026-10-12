#!/usr/bin/env bash
# Pobiera Common Voice Scripted Speech 27.0 (polski) z Mozilla Data Collective.
# Klucz API czyta z research/voice002/.env (MDC_API_KEY=...), plik jest poza gitem.
# Warunki zbioru trzeba wcześniej zaakceptować na stronie MDC.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dataset_id="cmu5wsxwp00e0nq07m8v0274j"
expected_sha="c9071ed61cf1836ccabf03a9a9cfeab827f8b7b4342b02a809dceb7597303e57"
out_dir="$root/data/public/common-voice-pl-27"
archive="$out_dir/cv-pl-27.tar.gz"

if [[ -z "${MDC_API_KEY:-}" && -f "$root/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; source "$root/.env"; set +a
fi
if [[ -z "${MDC_API_KEY:-}" ]]; then
  echo "Brak MDC_API_KEY (ustaw w $root/.env)" >&2
  exit 1
fi

mkdir -p "$out_dir"

if [[ ! -f "$archive.done" ]]; then
  response="$(curl -sS -X POST \
    -H "Authorization: Bearer $MDC_API_KEY" \
    -w '\n%{http_code}' \
    "https://mozilladatacollective.com/api/datasets/$dataset_id/download")"
  status="${response##*$'\n'}"
  body="${response%$'\n'*}"
  if [[ "$status" != "200" ]]; then
    echo "MDC API zwróciło HTTP $status: $body" >&2
    exit 1
  fi
  url="$(printf '%s' "$body" | python3 -c 'import sys,json; print(json.load(sys.stdin)["downloadUrl"])')"
  # -C - wznawia przerwane pobieranie.
  curl -fL --retry 5 -C - -o "$archive" "$url"

  actual_sha="$(shasum -a 256 "$archive" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "Zła suma SHA-256: $actual_sha (oczekiwano $expected_sha)" >&2
    exit 1
  fi
  touch "$archive.done"
fi

if [[ ! -f "$out_dir/.extracted" ]]; then
  tar -xzf "$archive" -C "$out_dir"
  touch "$out_dir/.extracted"
fi

echo "Gotowe: $out_dir"

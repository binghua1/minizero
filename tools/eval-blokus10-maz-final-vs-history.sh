#!/usr/bin/env bash
set -euo pipefail

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/eval-blokus10-maz-i300-vs-history.sh" "$@"

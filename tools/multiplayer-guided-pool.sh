#!/bin/bash

set -euo pipefail

repo_root=$(readlink -f "$(dirname "$0")/..")
exec "$repo_root/tools/tictacmo-jpsro.sh" "$@"

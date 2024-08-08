#!/bin/bash


# exit immediately on failing commands
set -euo pipefail
# to start api in docker container when container has started
exec python -m src.api.main

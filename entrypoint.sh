#!/bin/bash

# exit immediately on failing commands
set -euo pipefail

exec python -m src.api.main

#!/usr/bin/env bash
set -eu

RELEASE_REPO="pdomain/pdomain-ocr-training"

. "$(dirname "$0")/release-common.sh"
pdomain_release_main "$@"

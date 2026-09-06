#!/bin/bash
# Respect another benchmark's serving pause if this job failed preflight.
exec /usr/bin/python3 "$(dirname "$0")/decode-weight-reuse-wall-lifecycle.py" restore

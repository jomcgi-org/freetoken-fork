#!/bin/bash
# Restore the original model service after a detached wall-time job exits.
/usr/bin/sudo -n /usr/bin/systemctl stop astra-decode-weight-reuse-wall-server
/usr/bin/sudo -n /usr/bin/systemctl start freetoken-serve

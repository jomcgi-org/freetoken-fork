#!/bin/bash
# Recovery also runs if the detached benchmark driver exits unexpectedly.
/usr/bin/sudo -n /usr/bin/systemctl stop astra-concurrent-wall-server
/usr/bin/sudo -n /usr/bin/systemctl start freetoken-serve

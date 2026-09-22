# FreeToken fork agent instructions

## Review and merge authorization

The project owner explicitly authorized coding agents on 2026-09-22 to review
and merge their own pull requests for `jomcgi-org/freetoken-fork` without asking
for separate approval. This authorization is specific to this project.

Review the complete diff, run appropriate validation, and resolve blocking
findings before merging. Follow repository branch protections and required
checks. This permission does not waive validation requirements.

## Node-4 authorization

On 2026-09-22, the owner explicitly authorized node-4 as the FreeToken test
bench and approved operations there, including copying code to the machine,
running benchmarks, installing persistent systemd overrides, restarting the
serving service, and deploying validated revisions. These operations do not
require separate user approval. Retain rollback paths and verify service health
and inference after a deployment.

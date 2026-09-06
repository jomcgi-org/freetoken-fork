# Request slots in captured target verification

The wider target diagnostic previously retained rollback state through tensor
views into the request slot used during graph capture. Copying a different slot
into the model's input buffers would advance the new request while checkpoint
copies and restores still addressed the old one.

`SlotStateBindings` reads and writes the original state slabs through persistent
device indices. Linear state uses the linear slot; QSA pending state uses the
request-table slot. Full and compact checkpoints share these bindings, including
the compact checkpoint's initial recurrence and small retained prefix states.
The existing recurrent update function and its arithmetic remain unchanged.

The diagnostic opt-in is `FREETOKEN_TARGET_VERIFY_RELOCATABLE_STATE=1`, with width
three or five and the existing graph, serial-linear and checkpoint settings.
The graph copies incoming slot indices, positions, output locations and every
QSA row's addressing into persistent buffers. Host PLE staging reads the incoming
request history. Incompatible geometry, incomplete metadata, out-of-range host
request slots and pending lazy restores are rejected before buffer updates.
Graphs without explicit checkpoint bindings reject a change of request slot.

Index contents must remain unchanged from target submission through acceptance
or rollback. Bindings are installed once before warmup and graph capture; they
cannot be replaced afterward. The model and checkpoint index buffers must name
the same request. The scheduler must own all referenced state slots and KV pages.

Focused CPU checks cover every retained prefix at the supported widths, distinct
linear/request slots, untouched neighbours, activation-buffer reuse and staging
across noncontiguous page addresses. Explicit CUDA tests replay one captured
target and its restore graphs across different slots, including the real GDN
kernel with new inputs. Run those only with exclusive GPU ownership and automatic
original-serving recovery, after source staging and focused CPU validation.

With the relocation flag, the full-model cost harness captures each target graph
once and reuses it across the subsequent token windows. Qualification requires
one capture per variant and reuse in every later window, in addition to the
existing exact numerical and prefix-state checks. Other cost modes keep their
existing per-window capture protocol.

CPU address-copy tests alone do not qualify full-model graph reuse across requests
or page boundaries. Those model checks, a serving proposer and scheduler, task
completion checks and separate non-debug wall measurements remain required.
Gather/scatter operations may add component cost and graph-pool allocations.
This experiment claims no serving speedup and leaves serving startup unchanged.
Detailed model records and measured payloads stay private.

Validation completed with 118 focused Mac checks and 229 focused Linux checks
passing. The three exclusive CUDA checks passed, including the real GDN kernel
with changing slots and overwritten activation inputs. Full-model verification
then passed every required logit, token, committed-state and rejection-prefix
comparison while reusing each graph across adjacent windows in one allocated
page. Both experiments ended with verified original-serving recovery and a real
completion. Full-model relocation to other request slots and across page
boundaries remains unqualified, so serving integration is still pending.

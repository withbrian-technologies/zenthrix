# Memory planning

`analyze_lifetimes` computes a first and last execution position for every
tensor referenced by a graph node or marked as a graph output. The analysis
rejects invalid graphs and cyclic execution order.

`plan_memory` uses those intervals to assign aligned offsets in a single arena.
Intervals that no longer overlap may reuse space; active intervals are placed
after the current occupied range. The result reports each tensor allocation
and the peak arena size.

Device-specific allocators, zero-copy bindings, and hardware memory domains
remain future seams.

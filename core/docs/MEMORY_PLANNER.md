# Memory planning

`analyze_lifetimes` computes a first and last execution position for every
tensor referenced by a graph node or marked as a graph output. The analysis
rejects invalid graphs and cyclic execution order.

Marked graph outputs remain live through the end of execution so their arena
storage cannot be reused before the compiled result is consumed.

`plan_memory` uses those intervals to assign aligned offsets in a single arena.
Intervals that no longer overlap may reuse space; active intervals are placed
after the current occupied range. The result reports each tensor allocation
and the peak arena size.

Tensor shape and byte-size arithmetic is checked before planning. Alignment
padding, allocation ends, and arena-size calculations are also overflow-safe;
an unrepresentable plan raises `std::overflow_error` rather than returning
wrapped offsets.

Device-specific allocators, zero-copy bindings, and hardware memory domains
remain future seams.

# Compilation passes

`PassManager` executes passes in insertion order and records the names of
successful passes. A pass failure stops execution and leaves the graph in the
state produced by earlier successful passes.

Pass registration rejects null passes, passes with empty names, and duplicate
names. Every registered pass therefore has a unique, stable identifier for
execution records and failure diagnostics.

The current MVP includes:

- `ValidateGraphPass`, which gates the pipeline on graph invariants.
- `DeadCodeElimination`, which preserves graph outputs and removes nodes that
  cannot contribute to those outputs.

Future canonicalization, fusion, and layout passes should use the same
interface and must preserve deterministic node ordering.

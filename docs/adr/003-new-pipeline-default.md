# ADR 003: New Pipeline as Default

## Status
Accepted — 2025-09-01

## Context
The modern interfaces (`IDescriptorExtractor`, pooling strategies) coexisted with legacy runtime paths. The CLI used bridges to opt-in, increasing complexity.

## Decision
Make the new extractor path the only execution path in the CLI. Compute descriptors via wrappers and Schema v1 pooling overloads; matching via MatchingFactory; evaluation via TrueAveragePrecision.

## Consequences
- Cleaner CLI (`experiment_runner`) with a transparent flow.
- Easier to extend with new descriptors/wrappers.
- Legacy integration files and tests removed.


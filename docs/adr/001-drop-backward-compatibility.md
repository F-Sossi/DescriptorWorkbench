# ADR 001: Drop Backward Compatibility

## Status
Accepted — 2025-09-01

## Context
The codebase maintained legacy bridges (ConfigurationBridge, ProcessorBridge) to support old configuration flows. This added complexity and duplicated logic during the refactor to modern interfaces.

## Decision
Remove backward compatibility layers and run exclusively on the new pipeline using `IDescriptorExtractor`, Schema v1 configs, and modern pooling/matching.

## Consequences
- Simpler, clearer pipeline; single source of truth for behavior.
- Legacy bridge tests removed; tests focus on Schema v1 and core algorithms.
- Old paths are no longer available; configs must follow Schema v1.


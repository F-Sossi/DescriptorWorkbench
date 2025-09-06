# ADR 002: Schema v1 (Strict)

## Status
Accepted — 2025-09-01

## Context
Configs mixed legacy keys (e.g., `normalize`, `matching_threshold`) with new fields. Validation was partial, creating ambiguity and runtime errors.

## Decision
Adopt a single strict Schema v1 for YAML configs. Enforce required fields, ranges, and cross-field constraints in `YAMLConfigLoader`. Provide defaults and examples.

## Consequences
- Clear, actionable validation errors; fewer runtime surprises.
- All example configs updated; defaults added under `config/defaults/`.
- Migration keys are not accepted; docs reflect the new schema.


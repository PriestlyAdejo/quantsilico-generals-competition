# Phase 9F package migration plan (deferred physical move)

## Status
METADATA_CORRECTED; PHYSICAL_MIGRATION_DEFERRED

## Canonical ID
heuristic_v2f_plus_planner_terminal_fix / QS-P9F-PORTAL-V0

## Typos (not distinct)
- heuristic_v2f_plus_planner_terminal_force
- heuristic_v2f_plus_planner_terminal_form

## Current files (left in place temporarily)
Marked LEGACY_MISLABELLED in phase9f_package_registry_v2.json.
Original paths under dist/upload_ready/ preserved with hashes.

## Intended final layout (execute in final packaging window)
- dist/legacy_mislabelled_upload_ready/ — historical copies + this MIGRATION.md
- dist/windows_smoke_passed/ — Windows protocol smoke copies
- dist/upload_ready/ — empty until OFFICIAL_UPLOAD_READY
- dist/roles/ — role JSON with alias_of
- submission/packages/ — canonical builds

## Qualification
Windows handshake/EOF smoke != official upload ready.

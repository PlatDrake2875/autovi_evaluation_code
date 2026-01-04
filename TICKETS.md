# Branch: AIT-25/setup-config

## Overview
This branch handles project setup and configuration tasks, including adding dependencies and creating example configuration files for Stage 2 features.

**Note:** AIT-31 (scipy dependency) should be done early as other branches may need it.

## Tickets to Complete

### AIT-31: Add scipy dependency to pyproject.toml
**Status:** To Do
**Priority:** Do first - other code may need this

Add required dependencies to `pyproject.toml`:
- `scipy >= 1.11.0` (for `trim_mean` in robustness)

After updating, run:
```bash
uv sync
```

---

### AIT-25: Create example config YAML files for Stage 2 features
**Status:** To Do
**Depends on:** Robustness module structure (AIT-9) should exist

Create example configuration files in `configs/`:
- `robustness_example.yaml` - shows robustness config options
- `xai_example.yaml` - shows interpretability config options
- `full_trustworthy_example.yaml` - combines DP + robustness + XAI

---

## Suggested Order
1. AIT-31 (scipy) - do immediately, needed by other branches
2. AIT-25 (config examples) - do after robustness module is implemented

## Files to Create/Modify
- `pyproject.toml` (modify - add scipy)
- `configs/robustness_example.yaml` (new)
- `configs/xai_example.yaml` (new)
- `configs/full_trustworthy_example.yaml` (new)

## Notes
- The scipy dependency should be merged to main early so other branches can use it
- Config examples should reference the actual config dataclasses once implemented
- Consider merging AIT-31 separately before the rest of the work

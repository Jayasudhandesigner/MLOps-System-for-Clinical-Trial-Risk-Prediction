# Branch Strategy

**Enterprise-Grade Separation: Production vs Research**

---

## Branch Structure

### `main` - **Production Deployment**

**Purpose:** Deployment-ready code only

**Contains:**
- ✅ API server (`api/`)
- ✅ Core production code (`src/core/`)
- ✅ Configuration & requirements
- ✅ Documentation (knowledge base)
- ✅ Data versioning (DVC)
- ✅ Tests

**Does NOT contain:**
- ❌ Experimental code
- ❌ Jupyter notebooks
- ❌ Threshold tuning scripts
- ❌ Model comparison experiments
- ❌ Research artifacts

---

### `research` - **Experiments & Analysis**

**Purpose:** All learning, experimentation, and iteration

**Contains:**
- ✅ `src/experiments/` - Threshold tuning, feature experiments
- ✅ `scripts/tag_mlflow_runs.py` - MLflow management scripts
- ✅ Jupyter notebooks (if added)
- ✅ Ad-hoc analysis scripts
- ✅ Model comparison code

**Preserved for:**
- Audit trail
- Reproducibility
- Knowledge transfer
- Regulatory compliance

---

## Why This Separation?

### Production (main)

**Goal:** Clean, deployable, maintainable

**Benefits:**
- No clutter from experiments
- Clear deployment path
- Easy code review
- CI/CD friendly

**Example:** Google, Meta, pharmaceutical ML, fintech

### Research (research branch)

**Goal:** Preserve all learning

**Benefits:**
- Full experiment history
- Reproducible results
- Knowledge preserved
- Regulatory audit trail

---

## How to Use

### Deploy to Production

```bash
git checkout main
python api/main.py
```

Clean code, fast deployment, no confusion.

---

### Run Experiments

```bash
git checkout research
python src/experiments/threshold_tuning.py
```

All experimental tools available.

---

### Access Research Findings

Documentation in `main` **references** research:

**Example from `docs/05_MODEL_TUNING.md`:**

> **Model comparison, threshold tuning, and feature experiments**  
> were conducted in the `research` branch.
>
> See: `research` branch → `src/experiments/threshold_tuning.py`
>
> **Key findings:**
> - LightGBM achieved highest recall (0.8286)
> - Threshold tuning improved dropout detection 42%
> - Lower threshold (0.20) justified for high-risk trials

**Result:**
- ✅ Knowledge preserved in main
- ✅ Code isolated in research
- ✅ Audit trail intact
- ✅ Deployment clean

---

## Workflow: New Feature Development

### 1. Experiment in Research

```bash
git checkout research
# Create new experiment
python src/experiments/new_feature_test.py
# Iterate, test, validate
```

### 2. Graduate to Production

```bash
# Once validated, extract production code
git checkout main
# Add only production-ready code to src/core/
git add src/core/new_feature.py
git commit -m "feat: add validated new feature"
```

### 3. Update Documentation

```markdown
## New Feature

Feature validated in research branch (experiment ID: exp-123).
Implementation moved to production: src/core/new_feature.py
```

---

## Common Tasks

### View All Branches

```bash
git branch -a
```

Output:
```
* main
  research
```

### Switch Branches

```bash
git checkout research   # For experiments
git checkout main       # For deployment
```

### View Research Branch Without Switching

```bash
git show research:src/experiments/threshold_tuning.py
```

### Sync Knowledge (Research → Main)

**Update documentation only:**
```bash
git checkout main
# Edit docs/05_MODEL_TUNING.md
# Reference research findings
git commit -m "docs: update with research findings"
```

**Do NOT merge research into main** - they serve different purposes.

---

## What Files Live Where?

| File/Directory | main | research | Purpose |
|----------------|------|----------|---------|
| `api/` | ✅ | ✅ | API server |
| `src/core/` | ✅ | ✅ | Production code |
| `src/experiments/` | ❌ | ✅ | Experimental code |
| `docs/` | ✅ | ✅ | Documentation |
| `scripts/tag_mlflow_runs.py` | ❌ | ✅ | Research tooling |
| `requirements.txt` | ✅ | ✅ | Dependencies |
| `.dvc/` | ✅ | ✅ | Data versioning |
| `mlflow.db` | ✅ | ✅ | Experiment tracking |

---

## Enterprise Alignment

This approach matches:

| Company | Practice |
|---------|----------|
| **Google** | Clean prod branch + research forks |
| **Meta** | Prod services isolated from notebooks |
| **Pharma ML** | Strong branch isolation (regulatory) |
| **FinTech** | Strict separation (compliance) |

---

## Anti-Patterns (What NOT to Do)

❌ **Merge research into main** - Pollutes production code  
❌ **Deploy from research** - Unstable, cluttered  
❌ **Delete research** - Lose audit trail  
❌ **Mix experiments with production** - Hard to maintain  

✅ **Do this instead:**
- Keep branches separate
- Reference research in docs
- Graduate validated features
- Preserve all history

---

**Summary:** Production code in `main`, all learning in `research`, knowledge links both.

---

**Last Updated:** 2025-12-28  
**Strategy:** Enterprise-Grade Branch Separation  
**Status:** Active

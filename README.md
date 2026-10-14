# RootCause AI — Evidence-Based Autonomous Bug Investigation & Verification Agent

RootCause AI is an autonomous software debugging and verification system designed to investigate bugs, diagnose root causes using empirical repository evidence, generate minimal candidate patches, and independently verify fixes across three verification gates in isolated Git worktrees.

---

## 1. Problem Statement & Target Users

### The Problem
Software debugging and bug reproduction are among the most time-consuming engineering tasks. While Large Language Models can propose code modifications, they often suffer from:
- **Hallucinated diagnoses**: Fabricating non-existent file names, line numbers, or execution outcomes.
- **Unverified patches**: Proposing fixes that pass superficial inspections but break existing tests or fail to reproduce the reported bug.
- **Security risks**: Executing untrusted repository code directly in developers' working trees or leaking environment secrets.

### Target Users
- **Software Engineers & Pair Programmers**: Automating initial bug triage, root-cause localization, and reproduction test generation.
- **QA & Site Reliability Engineers (SREs)**: Verifying whether reported bugs reproduce deterministically and validating fix safety prior to deployment.
- **Platform & Security Teams**: Conducting safe, controlled code inspections with strict repository immutability and command sandboxing.

---

## 2. Proposed Solution & Main Workflow

RootCause AI enforces strict segregation between observed facts (Evidence) and model-generated conclusions (Hypotheses), operating through a six-stage workflow:

```
[Bug Ingestion] 
       │
       ▼
[Autonomous Investigation] ──(PathGuard & Subprocess Sandbox)──► [Empirical Evidence Collection]
       │
       ▼
[Evidence-Grounded Hypothesis] ──(Rule-Based Provenance Check)
       │
       ▼
[Candidate Patch & Regression Test Generation]
       │
       ▼
[Independent 3-Gate Verification] ──(Isolated Git Worktree)
       ├── Gate A: Baseline Bug Reproduction (Regression test fails on buggy baseline)
       ├── Gate B: Patched Regression Test (Regression test passes on patched worktree)
       └── Gate C: Regression Check (Pre-existing test suite passes with zero breaks)
       │
       ▼
[Human-Confirmed Patch Application] ──(Explicit confirmation + clean tree required)
```

1. **Ingestion**: Accepts target repository path, bug report description, and optional stack trace.
2. **Investigation**: Discovers code, reads files, inspects Git history, and executes targeted tests via sandboxed tools. All findings are recorded as immutable `Evidence` objects.
3. **Diagnosis**: Proposes root-cause hypotheses that **must strictly reference verified evidence IDs**.
4. **Patch Generation**: Produces a minimal unified diff and a dedicated reproduction test.
5. **Three-Gate Verification**:
   - **Gate A**: Confirms the reproduction test fails on the original baseline repository for the expected reason.
   - **Gate B**: Applies the diff in a separate, disposable Git worktree and confirms the reproduction test now passes.
   - **Gate C**: Executes the entire pre-existing test suite to ensure the patch introduced no regressions.
6. **Controlled Application**: The patch is never automatically applied to the original repository; application requires explicit human review and confirmation (`confirm=true`).

---

## 3. Architecture & Technology Stack

RootCause AI is built on a modular, decoupled Python architecture based strictly on the production codebase:

```
src/rootcause/
├── core/
│   ├── models.py           # Domain schemas (Evidence, Hypothesis, InvestigationState, CandidatePatch)
│   └── exceptions.py       # Domain exceptions (SecurityViolation, RateLimitError, MissingCredentialsError)
├── security/
│   ├── path_guard.py       # Traversal prevention, null-byte injection rejection, symlink escape checks
│   ├── sandbox.py          # Controlled subprocess execution, command allowlist, env secret scrubbing
│   └── worktree.py         # Isolated, disposable Git worktree manager
├── tools/
│   ├── file_tools.py       # Static file inspection and directory listing (never executes code)
│   ├── search_tools.py     # Static code & text search (literal and regex)
│   ├── git_tools.py        # Safe Git log, diff, commit, and status inspection
│   ├── trace_parser.py     # Multi-frame stack-trace parser and source-location extractor
│   └── test_runner.py      # Controlled pytest execution & explainable bug-reproduction matcher
├── orchestrator/
│   └── agent.py            # Multi-turn autonomous investigation orchestrator
├── db/
│   └── repository.py       # SQLite persistence layer with JSON serialization
├── api/
│   └── app.py              # FastAPI REST API with background task runner and sync options
├── llm/
│   ├── base.py             # BaseLLMProvider, LLMResponse with token usage & model tracking
│   ├── openai_provider.py  # Replaceable OpenAI-compatible adapter with 401/429 error handling
│   └── mock_provider.py    # Deterministic mock provider for local unit testing
├── verification/
│   ├── patch_engine.py     # Unified diff validation, security boundary checks, worktree patch applicator
│   └── verifier.py         # 3-Gate Independent Verification Pipeline & Patch Application Guard
└── benchmarks/
    ├── models.py           # BenchmarkCase, BenchmarkCaseResult, EvaluationMode, BenchmarkMetrics
    ├── dataset.py          # Curated 5-case benchmark registry
    ├── fixtures.py         # Deterministic Git repository generators and 5-point fixture validator
    ├── simulated_provider.py # Deterministic offline simulation provider for plumbing evaluation
    ├── tracking.py         # UsageTrackingProviderWrapper (tokens, models, retries, rate limits)
    ├── metrics.py          # Rule-based scorer with evidence verification & cost estimation
    └── runner.py           # CLI benchmark runner with secret scrubbing & fail-closed live mode
```

### Technology Stack
- **Language**: Python 3.10+ (tested on Python 3.12)
- **Data Validation & Schemas**: Pydantic v2
- **Web Framework**: FastAPI & Uvicorn (ASGI)
- **HTTP Client**: HTTPX
- **Database**: SQLite3
- **Test Engine & Sandboxing**: Pytest, Git CLI, Subprocess execution controls

---

## 4. Installation & Setup Instructions (Windows)

### Prerequisites
- Windows 10/11 or Windows Server
- Python 3.10 or higher
- Git installed and accessible in `PATH`

### Step 1: Clone or Open Repository
```powershell
cd C:\Users\kirti\.gemini\antigravity\scratch\rootcause-ai
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -e .
pip install pytest pytest-asyncio httpx uvicorn
```

### Step 4: Configure Environment Variables (Optional for Live Mode)
No API keys are required for offline development, automated tests, or simulated benchmarks. To enable live LLM evaluation:
```powershell
$env:ROOTCAUSE_LLM_API_KEY = "your-api-key"
# Optional overrides:
$env:ROOTCAUSE_LLM_BASE_URL = "https://api.openai.com/v1"
$env:ROOTCAUSE_LLM_MODEL = "gpt-4o-mini"
```

---

## 5. API Startup & Usage Examples

### Starting the API Server
Run the ASGI server via Uvicorn:
```powershell
$env:PYTHONPATH="src"
python -m uvicorn rootcause.api.app:app --host 127.0.0.1 --port 8000 --reload
```

### 1. Health Check
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```
Response:
```json
{
  "status": "ok",
  "service": "rootcause-ai",
  "version": "0.1.0"
}
```

### 2. Submit an Investigation (Synchronous)
```powershell
$body = @{
    repo_path = "C:\path\to\target\repo"
    bug_description = "Discount calculation applies 10% instead of 15% when total exceeds threshold"
    max_iterations = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations?sync=true" -Method Post -Body $body -ContentType "application/json"
```

### 3. Check Investigation Status & Evidence
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/<investigation_id>" -Method Get
```

### 4. Propose Patch & Verify
```powershell
# Propose candidate patch
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/<investigation_id>/propose-patch" -Method Post

# Execute 3-gate verification
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/<investigation_id>/verify" -Method Post
```

### 5. Apply Verified Patch to Original Repository
```powershell
# Requires explicit human confirmation (confirm=true)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/<investigation_id>/apply-patch?confirm=true" -Method Post
```

---

## 6. Test Execution Instructions

The automated test suite runs completely offline without requiring API keys or network access:

```powershell
$env:PYTHONPATH="src"
python -m pytest -v
```

### Actual Test Suite Structure
The repository contains **179 automated tests across 12 test modules**:
1. [`tests/test_benchmarks.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_benchmarks.py): 25 tests — Dataset registry, 5-point fixture validation, rule-based scorer, metric denominators, secret redaction, and offline runner.
2. [`tests/test_live_benchmark.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_live_benchmark.py): 20 tests — Live mode fail-closed checks, token aggregation, model tracking, rate-limit retries, cost estimation, and secret scrubbing.
3. [`tests/test_llm_provider.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_llm_provider.py): 8 tests — LLM interface abstractions, OpenAI-compatible adapter, tool-call formatting.
4. [`tests/test_orchestrator.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_orchestrator.py): 5 tests — Autonomous multi-turn investigation loop, evidence linkage, hypothesis submission.
5. [`tests/test_persistence_and_api.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_persistence_and_api.py): 7 tests — SQLite repository save/load/list, FastAPI sync/async endpoints.
6. [`tests/test_phase3_review.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_phase3_review.py): 6 tests — Phase 3 audit verification, honest insufficient evidence representation.
7. [`tests/test_repo_tools.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_repo_tools.py): 40 tests — File tools, search tools, git tools, test runner, explainable bug reproduction.
8. [`tests/test_security.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_security.py): 22 tests — Path traversal prevention, symlink escape rejection, subprocess command allowlisting.
9. [`tests/test_state_machine.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_state_machine.py): 7 tests — State transition validation and illegal transition rejections.
10. [`tests/test_trace_parser.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_trace_parser.py): 11 tests — Traceback extraction, multi-frame parsing, syntax errors.
11. [`tests/test_verification.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_verification.py): 24 tests — Unified diff validation, 3 verification gates, working tree immutability, patch application guard.
12. [`tests/test_worktree.py`](file:///C:/Users/kirti/.gemini/antigravity/scratch/rootcause-ai/tests/test_worktree.py): 4 tests — Isolated worktree creation, isolation guarantees, cleanup on error.

**Result**: **179 passed in ~99 seconds (0 failures, 0 skips, exit code 0)**.

---

## 7. Benchmark Commands & Results

### Supported Modes

| Mode | Command | Purpose |
| :--- | :--- | :--- |
| **Simulated** | `python -m rootcause.benchmarks.runner --mode simulated` | Validates orchestration, evidence extraction, and 3-gate verification plumbing offline using deterministic fixtures. |
| **Offline** | `python -m rootcause.benchmarks.runner --mode offline` | Functional alias for simulated mode. |
| **Live** | `python -m rootcause.benchmarks.runner --mode live --model gpt-4o-mini` | Evaluates autonomous LLM diagnosis and verification against a configured live provider. |

### Evaluation Integrity Statement
In compliance with Phase 5.1.1 and Phase 5.2 integrity rules:
- **Simulated offline results are strictly labeled `SIMULATED_OFFLINE`** and are never presented as autonomous LLM diagnostic performance.
- **Live results are strictly labeled `LIVE_PROVIDER`** and never fall back to simulation or inject ground-truth patches.

---

### Empirical Evaluation Results

#### A. Simulated Offline Benchmark (`--mode simulated`)
Executed across all 5 benchmark cases:
```
======================================================================
RootCause AI Benchmark Suite — Run ID: 4d2d8449-bf3d-4852-834c-0424bba0b01c
Evaluation Mode: SIMULATED_OFFLINE
Provider: MockLLMProvider (simulated offline)
Total Cases Evaluated: 5
Completed Investigations: 5/5 (100.0%)
Root Cause Matches: 5/5 (100.0%)
Gate A Reproduction: 5/5 (100.0%)
Gate B Patched Test: 5/5 (100.0%)
Gate C Full Suite: 5/5 (100.0%)
Verified Patches: 5/5 (100.0%)
Integrity Note: SIMULATED_OFFLINE: Results exercise orchestration, evidence gathering,
and verification plumbing using deterministic simulation. Not presented as autonomous LLM performance.
Results written to: benchmark_results.json
======================================================================
```

#### B. Live Benchmark Execution (`--mode live`)
When executed without credentials, live mode **fails closed**:
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
LIVE BENCHMARK NOT RUN: Missing credentials
To evaluate live LLM diagnostic accuracy, set one of:
  - ROOTCAUSE_LLM_API_KEY=<your-api-key>
  - OPENAI_API_KEY=<your-api-key>
Optional:
  - ROOTCAUSE_LLM_BASE_URL=<api-base-url>
  - ROOTCAUSE_LLM_MODEL=<model-name> (default: gpt-4o-mini)
Per Phase 5.2 integrity rules, live mode will not fall back to simulation.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

---

## 8. Security Measures, Known Limitations & Future Improvements

### Security Measures Implemented
1. **Repository Immutability**: All mutating commands, patches, and tests run exclusively in temporary Git worktrees. The original repository is never modified without explicit human confirmation.
2. **Path Containment (`PathGuard`)**: Resolves real paths; rejects traversal (`../`), null bytes, and symlink escapes.
3. **Controlled Subprocess Sandbox**: Restricts executables to `git`, `pytest`, and `python -m pytest`; disables shell interpretation (`shell=False`); strips secrets and tokens from environment variables.
4. **Secret Scrubbing**: Benchmark runner recursively redacts API keys, passwords, and Bearer tokens from error messages and serialized JSON.

### Known Limitations
1. **Host-Level Process Boundary**: While subprocess execution is heavily restricted by allowlisting and environment scrubbing, it runs with host user privileges. Untrusted code execution in `pytest` is not containerized at the kernel or microVM level.
2. **Single-Repository Scope**: Investigations currently target one repository at a time; cross-repository microservice dependencies are not automatically traced.
3. **Single-Language Focus**: Investigation tools and test runners are tailored for Python (`pytest`).

### Future Improvements
1. **Containerized Sandboxing**: Run test executions inside ephemeral Docker or gVisor microVMs for untrusted code execution.
2. **Multi-Language Support**: Extend test runner and stack-trace parser to JavaScript/TypeScript (Jest), Go (`go test`), and Rust (`cargo test`).
3. **Multi-File Complex Patching**: Support multi-hunk, multi-file refactorings and dependency version upgrades.

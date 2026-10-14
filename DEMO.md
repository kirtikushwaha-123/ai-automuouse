# RootCause AI — 3–5 Minute Evaluation Demonstration Script

This document provides an end-to-end walkthrough for demonstrating RootCause AI to evaluators, engineers, and stakeholders.

---

## Demonstration Overview

| Stage | Duration | Action | Key Takeaway |
| :--- | :---: | :--- | :--- |
| **1. Problem** | 30s | Explain the customer bug report | Debugging is slow; AI fixes often hallucinate or cause regressions. |
| **2. Ingestion** | 30s | Submit the bug to RootCause AI | Synchronous or asynchronous REST API submission. |
| **3. Evidence** | 45s | Inspect collected evidence & diagnosis | Hypotheses **must** cite empirical Evidence IDs gathered from real files. |
| **4. Patch Gen** | 45s | Generate candidate patch & test | Generates minimal unified diff and reproduction test. |
| **5. 3 Gates** | 60s | Execute 3-point independent verification | Rigorous verification: Gate A (reproduce), Gate B (fix), Gate C (no regression). |
| **6. Human Guard** | 30s | Confirm patch application | Human-in-the-loop: rejects unauthorized or dirty tree mutations. |
| **7. Benchmark** | 30s | Review benchmark integrity | Honest reporting separating simulated plumbing from live LLM evaluation. |

---

## Step 1: Explain the Customer Problem (30s)

> **Scenario**: An e-commerce customer reports that on orders of exactly \$100, the system applies a 0% discount instead of the advertised 10% discount.
>
> **Developer Dilemma**: Proposing a quick fix without reproduction risks breaking boundary conditions or creating silent regressions elsewhere in the checkout engine.

---

## Step 2: Submit a Reproducible Bug (30s)

Start the RootCause AI API server in Terminal 1:
```powershell
$env:PYTHONPATH="src"
python -m uvicorn rootcause.api.app:app --host 127.0.0.1 --port 8000
```

In Terminal 2, submit the bug investigation (using scenario `BM01_CONDITIONAL_LOGIC` as the target repo):
```powershell
# Create disposable target repo from the built-in benchmark fixture
python -c "from rootcause.benchmarks.fixtures import create_fixture_repository; from pathlib import Path; create_fixture_repository('BM01_CONDITIONAL_LOGIC', Path('demo_repo'))"

# Submit investigation synchronously
$body = @{
    repo_path = "$((Get-Item demo_repo).FullName)"
    bug_description = "Discount of 10% is not applied when cart total is exactly 100 dollars."
    max_iterations = 4
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations?sync=true" -Method Post -Body $body -ContentType "application/json"
$inv_id = $response.investigation_id
Write-Host "Investigation ID: $inv_id"
```

---

## Step 3: Show Repository Evidence & Proposed Root Cause (45s)

Query the investigation state to view the evidence-grounded diagnosis:
```powershell
$state = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/$inv_id" -Method Get
$state.facts | ForEach-Object { Write-Host "[$($_.id)] ($($_.evidence_type)) $($_.content)" }
Write-Host "`nDiagnosed Hypothesis:"
$state.hypotheses | ForEach-Object { Write-Host "$($_.title): $($_.description) (Cites: $($_.supporting_evidence_ids))" }
```

### Key Observation to Highlight
- The hypothesis is not an ungrounded guess. It explicitly references `supporting_evidence_ids` corresponding to real code lines extracted from `src/discount.py`:
  ```python
  if total > 100:  # Bug: should be total >= 100
      return total * 0.90
  ```

---

## Step 4: Generate Candidate Patch & Regression Test (45s)

Request a candidate patch and reproduction test:
```powershell
$patch_res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/$inv_id/propose-patch" -Method Post
Write-Host "Candidate Diff:"
Write-Host $patch_res.candidate_patch.diff
Write-Host "`nGenerated Regression Test:"
Write-Host $patch_res.regression_test.test_content
```

### Key Observation to Highlight
- The patch engine generates a minimal unified diff modifying only `src/discount.py`.
- It produces a dedicated reproduction test `tests/test_reproduction.py` that specifically tests the boundary case `calculate_discount(100.0) == 90.0`.

---

## Step 5: Demonstrate 3-Gate Independent Verification (60s)

Trigger independent verification in an isolated Git worktree:
```powershell
$verif = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/$inv_id/verify" -Method Post
Write-Host "Verification Status: $($verif.verified)"
Write-Host "Gate A (Baseline Fails as Expected): $($verif.gate_a.passed)"
Write-Host "Gate B (Patched Test Passes):        $($verif.gate_b.passed)"
Write-Host "Gate C (Full Suite No Regressions):  $($verif.gate_c.passed)"
```

### The Three Gates Explained
1. **Gate A**: Verifies the regression test actually fails on the unpatched code. If the test passed on the buggy baseline, it would prove the test is invalid.
2. **Gate B**: Applies the patch inside a disposable Git worktree and verifies the regression test now passes.
3. **Gate C**: Runs the entire pre-existing test suite to ensure the fix broke nothing else.

---

## Step 6: Show Human Confirmation Safeguard (30s)

Demonstrate safety controls preventing unauthorized modifications to the original repository:

### A. Rejected without Confirmation
```powershell
# Attempting to apply without confirm=true
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/$inv_id/apply-patch" -Method Post
} catch {
    Write-Host "Rejected as expected: $($_.Exception.Message)"
}
```
*Expected: HTTP 400 Bad Request — Explicit confirmation is required.*

### B. Approved with Confirmation
```powershell
# Apply with explicit human approval
$applied = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/investigations/$inv_id/apply-patch?confirm=true" -Method Post
Write-Host "Outcome: $($applied.message)"
```
*Outcome: The patch applies cleanly to the original repository.*

---

## Step 7: Explain Benchmark Results & Limitations Honestly (30s)

Conclude the demonstration by reviewing the evaluation framework:

### A. Run Simulated Offline Benchmark
```powershell
python -m rootcause.benchmarks.runner --mode simulated
```
- **Result**: 5/5 cases pass in deterministic offline simulation, proving that the multi-turn agent loop, evidence collection, and 3-gate verification engine operate reliably.

### B. Explain Live Evaluation & Fail-Closed Integrity
- Point out that when `--mode live` is executed without credentials, RootCause AI **fails closed**:
  ```
  LIVE BENCHMARK NOT RUN: Missing credentials
  ```
- **Integrity Rule**: RootCause AI never manufactures 100% scores by silently substituting ground-truth patches in live mode.

### Cleanup Demo Repository
```powershell
Remove-Item -Recurse -Force demo_repo
```

---

## Conclusion

RootCause AI eliminates the primary barrier to autonomous coding agents: **trust**. By grounding every hypothesis in empirical evidence and verifying every fix through independent gates in isolated worktrees, it guarantees that proposed patches actually fix the reported issue without breaking production code.

# Subordinating Conjunction Geometry & Discourse Vector Spaces

The geometry of subordinating conjunctions: can a single vector represent 2 ideas?

---

## Executive Summary & Research Notebook Progression

This repository investigates the latent geometric properties of subordinating conjunctions in transformer vector spaces. Specifically, we explore whether single dense vector representations (e.g., sentence transformer embeddings or LLM hidden states) can simultaneously preserve two distinct logical propositions (`Premise 1` and `Premise 2`) connected by causal, concessive, or conditional operators.

### Completed Notebooks & Key Findings

#### 1. Subordinating Conjunction Discourse Segmentation & Rhetorical Framing Analysis
- **Notebook File:** `1.subordinating_conjunction_discourse_segmentation.ipynb`
- **Target Feature:** Feature 02 — Subordinating Conjunction Density (`F02`)
- **Corpus & Sample Size:** PERSUADE 2.0 Lead Discourse Corpus ($N = 1,500$ total Lead elements, $N = 663$ $F02=1$ filtered samples).
- **LLM Engine:** Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) via `google-genai` SDK processed in 10-sample batches with persistent V2 disk caching (`llm_generation_cache_v2.json`).
- **Core Methodology:**
  1. Filtered the PERSUADE 2.0 dataset to isolate student introductory Lead elements exhibiting subordinating conjunction density ($F02 = 1$, prevalence $44.20\%$).
  2. Leveraged Gemini 3.1 Flash-Lite to segment each discourse element into four contiguous rhetorical categories:
     - `premise_1`: Primary proposition or main clause.
     - `premise_2`: Subordinate/dependent proposition containing evidence or constraints.
     - `conjunction`: The subordinating conjunction and related connective words (e.g., *because*, *although*, *if*, *since*).
     - `non_relevant`: Filler text, non-rhetorical introductory phrases, or trailing punctuation.
  3. Implemented a strict programmatic verification engine checking that every retained result contains non-empty `premise_1`, `premise_2`, and `conjunction` components. Fallbacks are strictly prohibited to prevent data pollution; unresolvable API errors raise explicit `RuntimeError` exceptions.
  4. Explicitly required the LLM prompt to return a clear `dismissal_reason` string whenever a sample is dismissed (`is_dismissed = True`).
  5. Built a dedicated validation cell (Section 5.1) that prints the complete LLM system prompt, raw response JSON string, and itemized parsed output for the first batch (Batch 1: 10 samples) prior to running the full processing pipeline.
  6. Executed the batch segmentation loop (Section 5.2) with per-batch validity logging reporting exact valid/invalid sample ratios after each batch request.
  7. Formatted 50 representative samples for visual human inspection audit.
  8. Exported final payloads to Google Drive (`/content/drive/MyDrive/persuade_data/`) and local fallback paths in CSV and JSON formats.
  9. Generated statistical metric dashboards displaying retention ratios, segment length distributions, conjunction term frequencies, and discourse effectiveness tier alignments.

# Human-Informed Neural Networks (HINN)

## Trust-Adaptive Learning with Expert Guidance

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/Status-Research-orange.svg)](#project-status)

<table>
<tr>
<td>

### Citation

If you use, adapt, extend, benchmark, or redistribute this software, please cite the associated publication:

**Mohammad (Behdad) Jamshidi, "Human-Informed Neural Networks (HINN): Trust-Adaptive Learning with Expert Guidance," *Human-Centric Intelligent Systems*, 2026.**

**Author:** Mohammad (Behdad) Jamshidi<br>
**Affiliation:** Faculty of Engineering and Information Technology, University of Technology Sydney, Sydney, Australia<br>
**Contact:** mohammad.jamshidi@alumni.uts.edu.au<br>
**BibTeX:** See the complete [Citation](#citation) section.

</td>
</tr>
</table>

Human-Informed Neural Networks (HINN) is a lightweight, dual-pathway learning framework for sequential decision problems in which human supervision is valuable but intermittent, noisy, or subject to drift. HINN combines:

- a deterministic recurrent neural network (RNN) that learns temporal structure from observed data;
- a variational Bayesian neural network (BNN) that models human-aligned supervisory signals and their uncertainty; and
- a time-sensitive trust function that changes the contribution of human guidance according to its recency and posterior confidence.

When human guidance is recent and confident, the human pathway contributes more strongly. When guidance becomes stale or uncertain, HINN moves toward data-driven learning. The repository includes a NumPy reference implementation, a controlled synthetic experiment, ablation baselines, extensive diagnostics, and a real-world AAPL volatility-risk case study.

> This is research software. It is not financial, medical, legal, or other professional advice, and it is not a production decision system without independent validation, governance, and domain-specific review.

## Contents

- [Why HINN?](#why-hinn)
- [Architecture](#architecture)
- [Repository features](#repository-features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Synthetic experiment](#synthetic-experiment)
- [Real-world AAPL case study](#real-world-aapl-case-study)
- [Mathematical overview](#mathematical-overview)
- [Python API](#python-api)
- [Repository layout](#repository-layout)
- [Reproducibility and outputs](#reproducibility-and-outputs)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Citation](#citation)
- [License](#license)

## Why HINN?

Purely data-driven objectives may not fully represent expert knowledge, policy requirements, behavioural preferences, or evolving operational constraints. Conversely, human feedback may be sparse, inconsistent, delayed, or wrong. HINN is designed for the space between these extremes: it incorporates human supervisory information without assuming that the information is continuously available or uniformly trustworthy.

The framework is intended for research on high-stakes sequential decision support, including financial risk assessment, clinical support, governance analytics, education, and policy-sensitive automation. Its central design goals are:

- **adaptive human-machine balance** rather than a fixed supervision weight;
- **uncertainty-aware guidance** through a Bayesian supervisory pathway;
- **graceful fallback** toward autonomous learning when supervision degrades;
- **transparent implementation** using NumPy and explicit recurrent updates; and
- **reproducible evaluation** through synthetic and real-data workflows.

## Architecture

```text
Sequential observations x(1:T)
          |
          v
  +---------------------+
  | Machine pathway     |
  | deterministic RNN   |-----> machine prediction p_t
  +---------------------+                 |
                                            v
Human labels y^H_t, mask m_t ---> +---------------------+
                                  | Human pathway       |
                                  | variational BNN     |
                                  +---------------------+
                                            |
                                  posterior prediction
                                  and uncertainty
                                            |
                         recency + confidence
                                            v
                                  trust coefficient
                                      lambda(t)
                                            |
                                            v
                               joint HINN optimisation
```

The machine prediction is also the interface through which the human-aligned loss propagates back into the recurrent representation. Human supervision therefore shapes the hidden-state dynamics through the joint gradient rather than through an assumed human target for an uninterpretable hidden state.

## Repository features

- Elman-style tanh RNN with explicit backpropagation through time.
- Mean-field variational BNN with reparameterised Monte Carlo sampling.
- Recency- and uncertainty-sensitive trust routing.
- Intermittent human-label masks, annotation noise, and policy drift.
- Data-only and human-only ablation baselines.
- Accuracy, macro-F1, calibration error, Brier score, multiclass AUC, posterior variance, KL divergence, and gradient diagnostics.
- Synthetic multi-asset regime-switching and GARCH-like experiment.
- Real AAPL OHLCV case study with chronological splitting and rolling volatility-risk targets.
- Publication-oriented plots, CSV metrics, and Markdown/LaTeX reports.

## Installation

### Requirements

- Python 3.9 or newer
- NumPy
- pandas
- scikit-learn
- Matplotlib
- seaborn

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Alternatively, install runtime dependencies without installing the package:

```bash
python -m pip install -r requirements.txt
```

Fresh ingestion from the external FNSPID dataset may require the optional Hugging Face `datasets` package and network access. The processed case-study inputs already included in this repository do not require re-ingestion.

## Quick start

Run the complete synthetic workflow after editable installation:

```bash
hinn-simulate
```

Equivalent entry points are:

```bash
python -m hinn
python main.py
```

Run the recommended real-world workflow:

```bash
python real_world_fnspid/run_hinn_aapl_rolling.py
```

The experiments are computational research workloads and may take time depending on hardware. They generate many figures and tables.

## Synthetic experiment

The synthetic workflow in `src/hinn/simulation.py` is the reference implementation. It creates approximately five years of daily multi-asset observations with:

- a persistent three-state latent market regime;
- correlated cross-asset shocks;
- GARCH-like volatility dynamics;
- ten model inputs derived from returns and technical indicators;
- low-, medium-, and high-risk targets based on forward realised volatility;
- human annotations with noise, gradual policy drift, and bursty availability; and
- a chronological 70/30 train-validation split.

The default experiment trains the full HINN, a data-only ablation, and a human-only ablation. It also generates diagnostics for training dynamics, calibration, uncertainty, trust routing, confusion matrices, robustness, posterior behaviour, and hyperparameter sensitivity.

### Redirecting generated outputs

By default, the synthetic runner writes into the current working directory. Set `HINN_OUTDIR` before launching it to keep outputs isolated.

Linux/macOS:

```bash
HINN_OUTDIR=outputs hinn-simulate
```

Windows PowerShell:

```powershell
$env:HINN_OUTDIR = "outputs"
hinn-simulate
```

Typical generated directories include `data/`, `figures/`, and `tables/`, together with a generated `REPORT.md`.

## Real-world AAPL case study

The `real_world_fnspid/` directory applies the shared HINN implementation to AAPL OHLCV data spanning 1980-12-12 through 2023-12-28. It contains 10,852 trading-day observations derived from the FNSPID research corpus, along with processed data and committed results for inspection.

Two runners are supplied:

| Runner | Target construction | Recommended use |
|---|---|---|
| `run_hinn_aapl.py` | Global volatility terciles | Historical comparison and diagnostic baseline |
| `run_hinn_aapl_rolling.py` | Trailing 252-day rolling volatility terciles | Recommended non-stationary case study |

The rolling workflow uses historical windows for locally adaptive targets and train-only scaling to reduce look-ahead leakage. Its human-supervision simulation includes 10% uniform label noise, gradual policy drift, and a seeded bursty mask with 46.6% coverage.

Run the recommended experiment and regenerate its plots:

```bash
python real_world_fnspid/run_hinn_aapl_rolling.py
python real_world_fnspid/generate_real_world_plots.py
python real_world_fnspid/make_case_figures.py
python real_world_fnspid/make_methodology_figures.py
```

Optional data-preparation commands are:

```bash
python real_world_fnspid/ingest_fnspid.py
python real_world_fnspid/fetch_aapl_price.py
python real_world_fnspid/filter_aapl_news.py
```

These preparation scripts may download or process external data. Review their configuration and the upstream dataset terms before running them.

### Committed case-study results

The repository includes result CSVs, figures, plot arrays, and a detailed report under `real_world_fnspid/results/`. These are research results for the included experimental configuration; they should not be interpreted as trading performance or evidence of production readiness.

## Mathematical overview

Let `x_t` be the input at time `t`, `h_t` the recurrent hidden state, and `p_t` the machine class-probability vector. The deterministic pathway computes:

```text
a_t = W_xh x_t + W_hh h_(t-1) + b_h
h_t = tanh(a_t)
z_t = W_hy h_t + b_y
p_t = softmax(z_t)
```

The human pathway places a mean-field Gaussian variational posterior over its parameters:

```text
q(theta | phi) = Normal(mu, diag(sigma^2))
theta = mu + sigma * epsilon,  epsilon ~ Normal(0, I)
```

For elapsed time `Delta t_h` since the last human signal and posterior predictive covariance `Cov_q`, the implementation defines:

```text
rho_t = exp(-gamma * Delta t_h)
c_t   = exp(-kappa * trace(Cov_q))

lambda(t) = lambda_min
            + (1 - lambda_min) * exp(-alpha * rho_t * c_t)
```

Here `lambda(t)` is the machine/data weight. Recent, confident human information makes `rho_t * c_t` large and moves `lambda(t)` toward its lower bound, increasing the relative human contribution. Stale or uncertain supervision moves `lambda(t)` toward `1`, restoring machine-dominant learning.

Conceptually, the joint objective is:

```text
L_HINN = sum_t [
    lambda(t) * L_data(t)
    + (1 - lambda(t)) * m_t * E_q[L_human(t)]
] + beta * KL(q(theta | phi) || p(theta))
```

where `m_t` gates unavailable human labels and the KL term regularises the variational posterior. Consult the associated paper and `src/hinn/simulation.py` for the complete formulation and implementation details.

## Python API

The package exports the principal components directly:

```python
from hinn import (
    BNNHuman,
    RNNMachine,
    expected_calibration_error,
    generate_financial_dataset,
    make_labels,
    train_hinn,
    trust_lambda,
)
```

The experiment runner is currently the most stable interface. The lower-level research API may evolve while the project remains in an early research stage.

## Repository layout

```text
.
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- main.py
|-- src/
|   `-- hinn/
|       |-- __init__.py
|       |-- __main__.py
|       `-- simulation.py
`-- real_world_fnspid/
    |-- run_hinn_aapl.py
    |-- run_hinn_aapl_rolling.py
    |-- ingest_fnspid.py
    |-- fetch_aapl_price.py
    |-- filter_aapl_news.py
    |-- generate_real_world_plots.py
    |-- make_case_figures.py
    |-- make_methodology_figures.py
    |-- data/
    `-- results/
```

## Reproducibility and outputs

- Random-number generators are seeded in the experimental code.
- Train-validation splits are chronological rather than randomly shuffled.
- The real-world rolling workflow uses train-only feature scaling.
- Synthetic generated outputs are ignored by Git because they can be recreated.
- Processed AAPL inputs and case-study outputs are committed for auditability.
- Hardware, dependency versions, and platform-level numerical differences may cause small result variations.

For rigorous comparisons, use an isolated environment, record the exact commit and package versions, and report all changes to seeds, data, targets, trust parameters, or training settings.

## Limitations and responsible use

- The code is a research prototype and has not been independently certified for safety-critical deployment.
- The included human labels are simulated proxies, not a clinical, regulatory, or financial expert panel.
- Bayesian predictive uncertainty is model-dependent and should not be treated as a guarantee of correctness.
- Graceful fallback reduces the influence of stale or uncertain supervision; it does not protect against every form of biased, adversarial, or confidently incorrect feedback.
- The AAPL study evaluates volatility-risk classification, not investment returns, execution costs, portfolio performance, or trading profitability.
- Results on one asset or synthetic environment do not establish generalisation to other domains.
- Production use requires independent validation, data-governance controls, monitoring, security review, fairness assessment, and accountable human oversight.

## Project status

HINN is research software under active development. Interfaces, experiment settings, and output formats may change. Issues and reproducible research contributions are welcome; please describe the environment, data provenance, seed, command, expected behaviour, and observed behaviour.

## Citation

If this repository contributes to a publication, thesis, report, software product, dataset, experiment, or derivative implementation, please cite:

> Mohammad (Behdad) Jamshidi, "Human-Informed Neural Networks (HINN): Trust-Adaptive Learning with Expert Guidance," *Human-Centric Intelligent Systems*, 2026.

```bibtex
@article{jamshidi2026hinn,
  author  = {Jamshidi, Mohammad (Behdad)},
  title   = {Human-Informed Neural Networks (HINN): Trust-Adaptive Learning with Expert Guidance},
  journal = {Human-Centric Intelligent Systems},
  year    = {2026},
  url     = {https://github.com/MBJamshidi/Human-Informed-Neural-Networks-HINN}
}
```

If a DOI, volume, issue, or final page range becomes available, use the publisher's final bibliographic record in preference to this provisional entry.

## License

The software is released under the [MIT License](LICENSE). Dataset files and upstream sources may be governed by separate terms; users are responsible for checking and complying with those terms.

---

> **Please cite the associated work:** All researchers, authors, developers, students, and organisations who use, adapt, extend, benchmark, or redistribute this repository are kindly requested to cite Mohammad (Behdad) Jamshidi, "Human-Informed Neural Networks (HINN): Trust-Adaptive Learning with Expert Guidance," *Human-Centric Intelligent Systems*, 2026.

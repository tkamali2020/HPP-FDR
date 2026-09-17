# HPP-FDRL — Reproducibility Package

**Hybrid Heart Patient Prioritization with Fuzzy Logic and Deep Reinforcement Learning for Fog–Cloud Task Scheduling**

This repository contains the implementation and experimental scripts required to reproduce the main results, ablation studies, sensitivity analysis, and fuzzy-rule export reported for the HPP-FDRL framework.

## Requirements

- Python 3.10 or later
- pip
- Recommended: a virtual environment

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Dataset

The experiments use the **UCI Heart Disease dataset**:

```text
data/heart_disease.csv
```

The dataset contains **303 patient records**.

Please ensure that the dataset file is available at the path above before running the experiments.

## Repository Structure

A typical repository structure is:

```text
HPP-FDRL/
├── data/
│   └── heart_disease.csv
├── config.py
├── run_comparison.py
├── ablation_runner.py
├── ablation_fuzzy_prioritization.py
├── ablation_drl_scheduling.py
├── sensitivity_analysis_priority_noise.py
├── export_fuzzy_rules.py
├── requirements.txt
└── README.md
```

## Main Experiments

### Static Scenario

The following command runs the full comparison using **30 independent random seeds**, consistent with the main experimental evaluation reported in the manuscript and Tables 6–7.

```bash
python run_comparison.py --scenario static --episodes 100 --seeds 30 --cases-per-episode 30 --out comparison_results.csv
```

The resulting file is:

```text
comparison_results.csv
```

### Dynamic Scenario

To evaluate the framework under the dynamic workload scenario:

```bash
python run_comparison.py --scenario dynamic --episodes 100 --seeds 30 --cases-per-episode 30 --out comparison_dynamic.csv
```

The resulting file is:

```text
comparison_dynamic.csv
```

## Ablation Studies

The ablation experiments use **5 independent seeds**, as reported in the Supplementary Material.

### Reward Ablation

```bash
python ablation_runner.py --scenario static --seeds 5 --episodes 50 --cases-per-episode 30 --out ablation_reward.csv
```

### Fuzzy Prioritization Ablation

```bash
python ablation_fuzzy_prioritization.py --scenario static --seeds 5 --episodes 50 --cases-per-episode 30 --out ablation_fuzzy.csv
```

### DRL Scheduling Ablation

```bash
python ablation_drl_scheduling.py --scenario static --seeds 5 --episodes 50 --cases-per-episode 30 --out ablation_drl.csv
```

## Sensitivity Analysis

The priority-noise sensitivity analysis evaluates the robustness of the prioritization component under perturbed priority values.

The analysis uses **5 independent seeds**:

```bash
python sensitivity_analysis_priority_noise.py --scenario static --seeds 5 --episodes 50 --cases-per-episode 30 --out sensitivity_priority_noise_results.csv
```

The resulting file is:

```text
sensitivity_priority_noise_results.csv
```

## Fuzzy Rule Export

The fuzzy prioritization module contains **324 fuzzy rules**.

To export the complete rule base:

```bash
python export_fuzzy_rules.py
```

The generated rule file can be used to inspect the complete fuzzy rule base independently of the scheduling experiments.

## Reward Configuration

The default reward weights are defined in `config.py`:

| Weight | Value |
|---|---:|
| `w1` | 0.20 |
| `w2` | 0.18 |
| `w3` | 0.22 |
| `w4` | 0.25 |
| `w5` | 0.15 |

These values correspond to the default configuration used for the reported experiments.

## Experimental Seeds

The reproducibility settings are divided according to the type of experiment:

| Experiment | Number of Seeds |
|---|---:|
| Main comparison | 30 |
| Ablation studies | 5 |
| Sensitivity analysis | 5 |
| Offloading analysis | 5 |

The main comparison uses **30 independent seeds**, while the ablation, sensitivity, and offloading analyses use **5 independent seeds**, consistent with the experimental protocol described in the Supplementary Material.

## Fog–Cloud Offloading

Cloud offloading is implemented as an **explicit action in the scheduling environment**.

The HPP-FDRL agent can therefore select between available execution/offloading actions during scheduling. Under the evaluated workloads, the learned policy exhibits a **fog-preferring behavior**, while retaining the ability to select cloud offloading when required by the environment.

This behavior should be interpreted within the evaluated workload and simulation settings rather than as a universal preference for fog execution.

## Reproducing the Reported Results

For a full reproduction of the main comparison:

1. Install the required dependencies.
2. Place `heart_disease.csv` in `data/`.
3. Verify the default configuration in `config.py`.
4. Run the static main experiment with 30 seeds.
5. Run the dynamic scenario if required.
6. Run the ablation and sensitivity experiments using their specified 5-seed configurations.
7. Export the 324 fuzzy rules using `export_fuzzy_rules.py`.
8. Compare the generated CSV files with the corresponding results reported in the manuscript and Supplementary Material.

## Notes

- Results may show small numerical differences across operating systems, Python versions, or dependency versions.
- The experiments are stochastic; the reported protocol uses independent random seeds to quantify variability.
- The dataset and experimental configuration should not be modified when attempting to reproduce the reported manuscript results.
- The commands above assume that they are executed from the repository root directory.

## Reproducibility Summary

| Component | Configuration |
|---|---|
| Dataset | UCI Heart Disease |
| Number of records | 303 |
| Python | 3.10+ |
| Main scenario | Static |
| Main seeds | 30 |
| Main episodes | 100 |
| Cases per episode | 30 |
| Ablation/sensitivity seeds | 5 |
| Ablation/sensitivity episodes | 50 |
| Fuzzy rules | 324 |
| Cloud offloading | Explicit environment action |

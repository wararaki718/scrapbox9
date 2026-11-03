# Interrupted Time Series Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable Python example that generates a small synthetic interrupted time series, estimates level and trend changes, and saves an explanatory plot.

**Architecture:** Keep the example self-contained in `main.py` with small functions for data generation, design-matrix construction, OLS fitting, result reporting, and plotting. List the three runtime libraries in `requirements.txt` and document the model and interpretation in the existing README.

**Tech Stack:** Python, NumPy, statsmodels OLS, Matplotlib.

---

## Files and responsibilities

- Create `math_sample/simulate-intempted-time-series/requirements.txt` with runtime dependencies only.
- Create `math_sample/simulate-intempted-time-series/main.py` as the executable analysis sample. It has no project-specific imports and uses deterministic data so repeated runs produce the same estimates.
- Update `math_sample/simulate-intempted-time-series/README.md` with the model, interpretation, setup, run command, generated image, and limitations.
- Do not add a separate test framework or test file; validate the small sample with direct assertions and an end-to-end execution.

## Task 1: Add dependencies and implement the analysis script

**Files:**
- Create: `math_sample/simulate-intempted-time-series/requirements.txt`
- Create: `math_sample/simulate-intempted-time-series/main.py`

- [ ] **Step 1: Declare the runtime dependencies**

Create `requirements.txt` with:

```text
numpy
statsmodels
matplotlib
```

- [ ] **Step 2: Implement the deterministic interrupted time series**

Create `main.py` with the following complete implementation. Use 24 monthly observations, with the intervention at index 12 (month 13). The residual pattern sums to zero and has zero linear trend within each segment, so the OLS coefficients recover the intended values while the observations still vary.

````python
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def generate_sample_data():
    time_index = np.arange(24, dtype=float)
    intervention_index = 12
    post_intervention = (time_index >= intervention_index).astype(float)
    time_after_intervention = np.maximum(time_index - intervention_index, 0)
    noise = np.tile(
        [-2, -1, 0, 1, 2, 0, 0, 2, 1, 0, -1, -2],
        2,
    )
    outcome = (
        20
        + 1.5 * time_index
        + 8 * post_intervention
        + 0.7 * time_after_intervention
        + noise
    )
    return time_index, outcome, intervention_index


def build_design_matrix(time_index, intervention_index):
    post_intervention = (time_index >= intervention_index).astype(float)
    time_after_intervention = np.maximum(time_index - intervention_index, 0)
    return np.column_stack(
        (
            np.ones_like(time_index),
            time_index,
            post_intervention,
            time_after_intervention,
        )
    )


def fit_interrupted_time_series(outcome, design_matrix):
    return sm.OLS(outcome, design_matrix).fit()


def print_results(model):
    parameter_names = (
        "Intercept (beta_0)",
        "Baseline slope (beta_1)",
        "Level change (beta_2)",
        "Slope change (beta_3)",
    )
    for name, estimate in zip(parameter_names, model.params):
        print(f"{name}: {estimate:.3f}")

    baseline_slope = model.params[1]
    post_intervention_slope = baseline_slope + model.params[3]
    print(f"Baseline slope: {baseline_slope:.3f}")
    print(f"Post-intervention slope (beta_1 + beta_3): {post_intervention_slope:.3f}")

    confidence_intervals = model.conf_int()
    for index, name in ((2, "Level change (beta_2)"), (3, "Slope change (beta_3)")):
        lower, upper = confidence_intervals[index]
        print(
            f"{name}: p={model.pvalues[index]:.4f}, "
            f"95% CI=[{lower:.3f}, {upper:.3f}]"
        )


def plot_results(time_index, outcome, intervention_index, model):
    months = time_index + 1

    plt.figure(figsize=(10, 5))
    plt.plot(months, outcome, marker="o", label="Observed")
    plt.plot(months, model.fittedvalues, linestyle="--", label="Fitted")
    plt.axvline(
        intervention_index + 1,
        color="red",
        linestyle=":",
        label="Intervention",
    )
    plt.xlabel("Month")
    plt.ylabel("Outcome")
    plt.title("Interrupted Time Series")
    plt.legend()
    plt.tight_layout()
    plt.savefig("interrupted_time_series.png", dpi=150)
    plt.close()


def main():
    time_index, outcome, intervention_index = generate_sample_data()
    design_matrix = build_design_matrix(time_index, intervention_index)
    model = fit_interrupted_time_series(outcome, design_matrix)
    print_results(model)
    plot_results(time_index, outcome, intervention_index, model)


if __name__ == "__main__":
    main()
````

- [ ] **Step 3: Install the declared dependencies**

Run from the sample directory:

```bash
python -m pip install -r requirements.txt
```

Expected: pip completes successfully with NumPy, statsmodels, and Matplotlib available.

- [ ] **Step 4: Check the functions and coefficient recovery**

From the sample directory, run:

```bash
python - <<'PY'
import numpy as np
from main import (
    build_design_matrix,
    fit_interrupted_time_series,
    generate_sample_data,
)

time_index, outcome, intervention_index = generate_sample_data()
design_matrix = build_design_matrix(time_index, intervention_index)
model = fit_interrupted_time_series(outcome, design_matrix)

assert len(time_index) == 24
assert len(outcome) == 24
assert intervention_index == 12
assert design_matrix.shape == (24, 4)
np.testing.assert_allclose(model.params, [20.0, 1.5, 8.0, 0.7], atol=1e-10)
print("data, design matrix, and coefficients: OK")
PY
```

Expected: `data, design matrix, and coefficients: OK`.

## Task 2: Document execution and interpretation, then verify end to end

**Files:**
- Modify: `math_sample/simulate-intempted-time-series/README.md`

- [ ] **Step 1: Document the sample**

Replace the current one-line README with:

````markdown
# Interrupted Time Series Simulation

介入の前後で時系列の水準と傾きがどう変化したかを、分割回帰で確認するサンプルです。月次24件の合成データをコード内で決定論的に作成します。

## 分析モデル

次のOLSモデルを推定します。

```text
Y_t = beta_0 + beta_1 * t + beta_2 * post_t + beta_3 * time_after_t + error_t
```

- `beta_0`: 介入前の切片
- `beta_1`: 介入前の傾き
- `beta_2`: 介入時点での水準変化
- `beta_3`: 介入後の傾きの変化
- 介入後の傾き: `beta_1 + beta_3`

実行すると、係数、介入による水準・傾き変化のp値と95%信頼区間を表示し、観測値・推定値・介入時点を `interrupted_time_series.png` に保存します。

## 実行方法

```shell
python -m pip install -r requirements.txt
python main.py
```

## 注意点

これは手法を説明するための合成データ例です。単純なOLSのため、自己相関、季節性、交絡要因は調整していません。係数だけで介入の因果効果が確定するわけではありません。
````

- [ ] **Step 2: Run the example end to end**

Run from the sample directory:

```bash
python main.py
```

Expected: the console reports coefficients approximately equal to \(\beta_0=20\), \(\beta_1=1.5\), \(\beta_2=8\), and \(\beta_3=0.7\); the baseline and post-intervention slopes are approximately 1.5 and 2.2; and `interrupted_time_series.png` is created.

- [ ] **Step 3: Confirm documentation and changed-file formatting**

From the sample directory, run:

```bash
python - <<'PY'
from pathlib import Path

paths = (Path("README.md"), Path("main.py"), Path("requirements.txt"))
for path in paths:
    text = path.read_text()
    assert text.endswith("\n"), f"{path} must end with a newline"
    assert all(line == line.rstrip() for line in text.splitlines()), (
        f"{path} contains trailing whitespace"
    )

readme = Path("README.md").read_text()
script = Path("main.py").read_text()
assert "python main.py" in readme
assert "interrupted_time_series.png" in readme
assert 'plt.savefig("interrupted_time_series.png"' in script
print("README and sample files: OK")
PY
```

Expected: `README and sample files: OK`.

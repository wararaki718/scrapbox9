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

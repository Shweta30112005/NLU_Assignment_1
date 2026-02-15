import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_theme(style="whitegrid")    # theme = whitegrid
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120
})


def generate_plots(project_root):

    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(results_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    overall_df = pd.read_csv(os.path.join(results_dir, "all_results.csv"))
    confusion_df = pd.read_csv(os.path.join(results_dir, "confusion_matrices.csv"))

    # Macro F1 Bar Chart
    sorted_df = overall_df.sort_values("f1_macro", ascending=False)

    plt.figure()
    sns.barplot(data=sorted_df, x="experiment", y="f1_macro", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Macro F1 Score Across Experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "macro_f1_comparison.png"), dpi=300)
    plt.close()

    # Accuracy vs Training Time
    plt.figure()
    sns.scatterplot(
        data=overall_df,
        x="training_time_sec",
        y="accuracy",
        hue="model",
        style="feature",
        s=100
    )
    plt.title("Accuracy vs Training Time")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuracy_vs_training_time.png"), dpi=300)
    plt.close()

    # Average Metrics by Feature Type
    feature_group = overall_df.groupby("feature")[["accuracy", "f1_macro"]].mean().reset_index()
    feature_melted = feature_group.melt(id_vars="feature", var_name="metric", value_name="score")

    plt.figure()
    sns.barplot(data=feature_melted, x="feature", y="score", hue="metric", palette="Set2")
    plt.title("Average Performance by Feature Type")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_comparison.png"), dpi=300)
    plt.close()

    #  F1 Trend Line across all models
    overall_df_sorted = overall_df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    plt.figure()
    sns.lineplot(data=overall_df_sorted, x=overall_df_sorted.index, y="f1_macro", marker="o")

    best_index = 0
    plt.scatter(best_index, overall_df_sorted.loc[best_index, "f1_macro"], color="red", s=150, label="Best Model")

    plt.xticks(ticks=range(len(overall_df_sorted)), labels=overall_df_sorted["experiment"], rotation=45, ha="right")
    plt.title("F1 Score Trend Across Experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "f1_trend.png"), dpi=300)
    plt.close()

    # Confusion Matrix Heatmap (Best Model)
    best_experiment = overall_df_sorted.iloc[0]["experiment"]
    best_cm_row = confusion_df[confusion_df["experiment"] == best_experiment].iloc[0]

    cm_matrix = np.array([
        [best_cm_row["politics_pred_politics"], best_cm_row["politics_pred_sport"]],
        [best_cm_row["sport_pred_politics"], best_cm_row["sport_pred_sport"]]
    ])

    plt.figure()
    sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Politics", "Sport"],
                yticklabels=["Politics", "Sport"])
    plt.title(f"Confusion Matrix - {best_experiment}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "best_confusion_matrix.png"), dpi=300)
    plt.close()

    print("\nAll visualizations generated successfully.")
    print(f"Saved in: {plots_dir}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_plots(root)

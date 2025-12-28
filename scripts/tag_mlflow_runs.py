"""
Tag best MLflow runs from threshold tuning experiment
"""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()

# Get threshold tuning experiment
exp = client.get_experiment_by_name("threshold_tuning_lightgbm")

# Find best recall run (threshold 0.20)
runs_by_recall = client.search_runs(
    exp.experiment_id,
    order_by=["metrics.recall DESC"],
    max_results=1
)

best_recall_run = runs_by_recall[0]
print(f"✅ Best Recall Run:")
print(f"   Run ID: {best_recall_run.info.run_id}")
print(f"   Threshold: {best_recall_run.data.params.get('decision_threshold')}")
print(f"   Recall: {best_recall_run.data.metrics.get('recall'):.4f}")

# Tag the run
client.set_tag(best_recall_run.info.run_id, "production", "true")
client.set_tag(best_recall_run.info.run_id, "optimization_goal", "max_recall")
client.set_tag(best_recall_run.info.run_id, "validated", "2025-12-28")
client.set_tag(best_recall_run.info.run_id, "deployment_status", "ready")

print(f"\n🏷️ Tagged run with: production=true, optimization_goal=max_recall")

# Find best F1 run (threshold 0.25)
runs_by_f1 = client.search_runs(
    exp.experiment_id,
    order_by=["metrics.f1 DESC"],
    max_results=1
)

best_f1_run = runs_by_f1[0]
print(f"\n✅ Best F1 Run:")
print(f"   Run ID: {best_f1_run.info.run_id}")
print(f"   Threshold: {best_f1_run.data.params.get('decision_threshold')}")
print(f"   F1 Score: {best_f1_run.data.metrics.get('f1'):.4f}")

# Tag the run
client.set_tag(best_f1_run.info.run_id, "alternative", "true")
client.set_tag(best_f1_run.info.run_id, "optimization_goal", "balanced_f1")
client.set_tag(best_f1_run.info.run_id, "validated", "2025-12-28")

print(f"\n🏷️ Tagged run with: alternative=true, optimization_goal=balanced_f1")

print(f"\n✅ MLflow tagging complete!")

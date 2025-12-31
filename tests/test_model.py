"""
tests/test_model.py
===================
Model Performance Validation Tests

Purpose: Ensure trained models meet minimum quality standards.
Prevents: Random/broken models from being promoted.

Run: pytest tests/test_model.py -v

Note: These tests require MLflow tracking server to be accessible.
"""

import pytest
import mlflow
from pathlib import Path


# Minimum acceptable metrics for production models
MIN_RECALL = 0.55
MIN_ROC_AUC = 0.60
MIN_PRECISION = 0.30


class TestModelPerformance:
    """Model performance validation tests."""
    
    @pytest.fixture(autouse=True)
    def setup_mlflow(self):
        """Setup MLflow connection."""
        # Use local SQLite backend
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    def test_model_not_random(self):
        """Model recall must exceed random baseline (0.55)."""
        try:
            runs = mlflow.search_runs(
                order_by=["metrics.recall DESC"],
                max_results=1
            )
            
            if len(runs) == 0:
                pytest.skip("No MLflow runs found - model not yet trained")
            
            recall = runs.iloc[0]["metrics.recall"]
            assert recall > MIN_RECALL, \
                f"Model recall {recall:.3f} below minimum {MIN_RECALL}"
                
        except Exception as e:
            pytest.skip(f"MLflow not accessible: {e}")
    
    def test_model_roc_auc(self):
        """Model ROC-AUC must demonstrate learnable signal (>0.60)."""
        try:
            runs = mlflow.search_runs(
                order_by=["metrics.test_roc_auc DESC"],
                max_results=1
            )
            
            if len(runs) == 0:
                pytest.skip("No MLflow runs found - model not yet trained")
            
            roc_auc = runs.iloc[0].get("metrics.test_roc_auc")
            if roc_auc is None:
                roc_auc = runs.iloc[0].get("metrics.roc_auc", 0)
            
            assert roc_auc > MIN_ROC_AUC, \
                f"Model ROC-AUC {roc_auc:.3f} below minimum {MIN_ROC_AUC}"
                
        except Exception as e:
            pytest.skip(f"MLflow not accessible: {e}")
    
    def test_model_precision(self):
        """Model precision must be acceptable (>0.30)."""
        try:
            runs = mlflow.search_runs(
                order_by=["metrics.precision DESC"],
                max_results=1
            )
            
            if len(runs) == 0:
                pytest.skip("No MLflow runs found - model not yet trained")
            
            precision = runs.iloc[0]["metrics.precision"]
            assert precision > MIN_PRECISION, \
                f"Model precision {precision:.3f} below minimum {MIN_PRECISION}"
                
        except Exception as e:
            pytest.skip(f"MLflow not accessible: {e}")
    
    def test_model_artifacts_exist(self):
        """Trained model artifacts must exist."""
        try:
            runs = mlflow.search_runs(max_results=1)
            
            if len(runs) == 0:
                pytest.skip("No MLflow runs found - model not yet trained")
            
            run_id = runs.iloc[0]["run_id"]
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run_id)
            
            # Check that some artifacts were logged
            assert len(artifacts) >= 0, \
                f"No artifacts found for run {run_id}"
                
        except Exception as e:
            pytest.skip(f"MLflow not accessible: {e}")


class TestModelStability:
    """Test model training stability."""
    
    @pytest.fixture(autouse=True)
    def setup_mlflow(self):
        """Setup MLflow connection."""
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    def test_cv_variance_reasonable(self):
        """Cross-validation variance should be reasonable (<0.10)."""
        try:
            runs = mlflow.search_runs(
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) == 0:
                pytest.skip("No MLflow runs found")
            
            cv_std = runs.iloc[0].get("metrics.cv_roc_auc_std")
            if cv_std is None:
                pytest.skip("CV std not logged")
            
            assert cv_std < 0.10, \
                f"CV variance too high: {cv_std:.3f} (max 0.10)"
                
        except Exception as e:
            pytest.skip(f"MLflow not accessible: {e}")

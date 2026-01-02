"""
tests/test_risk_consistency.py
==============================
RISK BAND CONSISTENCY TESTS

Ensures:
1. Same input → Same risk_band (determinism)
2. Known high-risk input → HIGH or CRITICAL
3. Risk thresholds are correctly applied

Run: pytest tests/test_risk_consistency.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.risk_bands import (
    get_risk_band,
    get_risk_assessment,
    compute_input_hash,
    RiskBand,
    THRESHOLD_VERSION
)


class TestRiskBandDeterminism:
    """Test that risk banding is deterministic."""
    
    def test_same_input_same_risk_band(self):
        """CRITICAL: Same input must produce same risk_band."""
        # Test score in the middle of HIGH band
        test_score = 0.65
        
        # Call multiple times
        result1 = get_risk_band(test_score)
        result2 = get_risk_band(test_score)
        result3 = get_risk_band(test_score)
        
        assert result1 == result2 == result3, \
            f"Risk band not deterministic: {result1}, {result2}, {result3}"
    
    def test_assessment_determinism(self):
        """Full assessment must be deterministic."""
        test_score = 0.55
        
        assessment1 = get_risk_assessment(test_score)
        assessment2 = get_risk_assessment(test_score)
        
        assert assessment1["risk_band"] == assessment2["risk_band"]
        assert assessment1["recommended_action"] == assessment2["recommended_action"]
        assert assessment1["intervention_cost"] == assessment2["intervention_cost"]
    
    def test_input_hash_determinism(self):
        """Input hash must be deterministic."""
        test_input = {
            "patient_id": "P-1234",
            "age": 65,
            "gender": "Female",
            "treatment_group": "Placebo"
        }
        
        hash1 = compute_input_hash(test_input)
        hash2 = compute_input_hash(test_input)
        
        assert hash1 == hash2, f"Hash not deterministic: {hash1} vs {hash2}"


class TestRiskBandThresholds:
    """Test risk band threshold boundaries."""
    
    def test_critical_threshold(self):
        """Score >= 0.75 must be CRITICAL."""
        assert get_risk_band(0.75) == RiskBand.CRITICAL
        assert get_risk_band(0.80) == RiskBand.CRITICAL
        assert get_risk_band(0.99) == RiskBand.CRITICAL
        assert get_risk_band(1.00) == RiskBand.CRITICAL
    
    def test_high_threshold(self):
        """Score >= 0.55 and < 0.75 must be HIGH."""
        assert get_risk_band(0.55) == RiskBand.HIGH
        assert get_risk_band(0.60) == RiskBand.HIGH
        assert get_risk_band(0.74) == RiskBand.HIGH
        # Boundary check
        assert get_risk_band(0.749999) == RiskBand.HIGH
    
    def test_medium_threshold(self):
        """Score >= 0.35 and < 0.55 must be MEDIUM."""
        assert get_risk_band(0.35) == RiskBand.MEDIUM
        assert get_risk_band(0.45) == RiskBand.MEDIUM
        assert get_risk_band(0.54) == RiskBand.MEDIUM
    
    def test_low_threshold(self):
        """Score < 0.35 must be LOW."""
        assert get_risk_band(0.00) == RiskBand.LOW
        assert get_risk_band(0.10) == RiskBand.LOW
        assert get_risk_band(0.34) == RiskBand.LOW


class TestHighRiskPatients:
    """Integration test: Known high-risk patterns → HIGH or CRITICAL."""
    
    def test_high_risk_score_returns_high_or_critical(self):
        """Known high-risk score must return HIGH or CRITICAL."""
        high_risk_scores = [0.65, 0.72, 0.78, 0.85, 0.92]
        
        for score in high_risk_scores:
            band = get_risk_band(score)
            assert band in [RiskBand.HIGH, RiskBand.CRITICAL], \
                f"Score {score} should be HIGH or CRITICAL, got {band}"
    
    def test_threshold_version_present(self):
        """Assessment must include threshold version for auditability."""
        assessment = get_risk_assessment(0.50)
        
        assert "threshold_version" in assessment
        assert assessment["threshold_version"] == THRESHOLD_VERSION


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_score_clamping_high(self):
        """Scores > 1.0 must be clamped to 1.0."""
        assert get_risk_band(1.5) == RiskBand.CRITICAL
        assert get_risk_band(100.0) == RiskBand.CRITICAL
    
    def test_score_clamping_low(self):
        """Scores < 0.0 must be clamped to 0.0."""
        assert get_risk_band(-0.5) == RiskBand.LOW
        assert get_risk_band(-100.0) == RiskBand.LOW
    
    def test_exact_boundary_values(self):
        """Test exact boundary values."""
        # These are the EXACT thresholds
        assert get_risk_band(0.35) == RiskBand.MEDIUM  # >= 0.35
        assert get_risk_band(0.55) == RiskBand.HIGH    # >= 0.55
        assert get_risk_band(0.75) == RiskBand.CRITICAL  # >= 0.75
        
        # Just below thresholds
        assert get_risk_band(0.349) == RiskBand.LOW
        assert get_risk_band(0.549) == RiskBand.MEDIUM
        assert get_risk_band(0.749) == RiskBand.HIGH


class TestAssessmentStructure:
    """Test that assessments have correct structure."""
    
    def test_assessment_has_all_fields(self):
        """Assessment must have all required fields."""
        required_fields = [
            "risk_band",
            "recommended_action",
            "intervention_cost",
            "action_description",
            "priority",
            "threshold_version"
        ]
        
        assessment = get_risk_assessment(0.50)
        
        for field in required_fields:
            assert field in assessment, f"Missing field: {field}"
    
    def test_cost_increases_with_risk(self):
        """Higher risk bands should have higher intervention costs."""
        low_cost = get_risk_assessment(0.20)["intervention_cost"]
        medium_cost = get_risk_assessment(0.45)["intervention_cost"]
        high_cost = get_risk_assessment(0.65)["intervention_cost"]
        critical_cost = get_risk_assessment(0.85)["intervention_cost"]
        
        assert low_cost < medium_cost < high_cost < critical_cost, \
            f"Costs should increase with risk: {low_cost} < {medium_cost} < {high_cost} < {critical_cost}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

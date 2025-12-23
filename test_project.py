"""Test script to verify the MLOps project works correctly."""
from src.ingest import load_data, REQUIRED_COLUMNS
from src.utils import get_logger

logger = get_logger('test')

def test_ingestion():
    """Test the data ingestion module."""
    logger.info("Testing ingestion module...")
    
    # Test 1: Load data
    data = load_data('data/raw/clinical_trials.csv')
    logger.info(f"✓ Data loaded successfully: {data.shape}")
    
    # Test 2: Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"
    logger.info(f"✓ All required columns present: {REQUIRED_COLUMNS}")
    
    # Test 3: Check for null patient_ids
    assert data["patient_id"].isnull().sum() == 0
    logger.info("✓ No null patient_ids")
    
    # Test 4: Check dropout values are binary
    assert data["dropout"].isin([0, 1]).all()
    logger.info("✓ Dropout values are binary (0 or 1)")
    
    # Test 5: Data statistics
    logger.info(f"\nData Statistics:")
    logger.info(f"  - Total patients: {len(data)}")
    logger.info(f"  - Dropout rate: {data['dropout'].mean():.2%}")
    logger.info(f"  - Age range: {data['age'].min()} - {data['age'].max()}")
    logger.info(f"  - Average visits completed: {data['visits_completed'].mean():.2f}")
    
    logger.info("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_ingestion()

import pandas as pd

# Check processed data
df = pd.read_csv('data/processed/clinical_trials_dropout.csv')

print("="*70)
print("V5 DATA VERIFICATION")
print("="*70)
print(f"Shape: {df.shape}")
print(f"Dropout rate: {df['dropout'].mean():.2%}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample data:")
print(df.head())

# Check if it has the archetype column (indicates V5 data)
if 'archetype' in df.columns:
    print("\n✅ This IS V5 data (has archetype column)")
    print(f"\nArchetype distribution:")
    print(df['archetype'].value_counts())
else:
    print("\n❌ This is NOT V5 data (missing archetype column)")

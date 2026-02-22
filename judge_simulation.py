"""
Simulates EXACTLY what the hackathon judge container does.
This will expose the exact runtime error.
"""
import sys
import traceback
import pandas as pd

print("=" * 60)
print("JUDGE SIMULATION — Full Pipeline Test")
print("=" * 60)

# Load test proxy (use train data, drop target to simulate test.csv)
print("\n[1] Loading test data proxy...")
df_raw = pd.read_csv("train.csv").drop(columns=["Purchased_Coverage_Bundle"])
print(f"    Shape: {df_raw.shape}, Columns: {list(df_raw.columns[:5])}...")

# Import solution exactly as the judge would
print("\n[2] Importing solution.py...")
try:
    from solution import preprocess, load_model, predict
    print("    ✅ Import OK")
except Exception as e:
    print(f"    ❌ IMPORT FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 1: preprocess
print("\n[3] Running preprocess(df)...")
try:
    df_processed = preprocess(df_raw)
    print(f"    ✅ OK — shape: {df_processed.shape}")
    print(f"    Dtypes with issues:")
    for col, dtype in df_processed.dtypes.items():
        if str(dtype) not in ['float64','int64','int32','object','bool']:
            print(f"      ⚠ {col}: {dtype}")
    # Check for any Categorical columns (these can break sklearn)
    cat_cols = [c for c in df_processed.columns if str(df_processed[c].dtype) == 'category']
    if cat_cols:
        print(f"    ❌ CATEGORICAL COLUMNS FOUND (will break ColumnTransformer!): {cat_cols}")
    else:
        print(f"    ✅ No Categorical dtype columns")
except Exception as e:
    print(f"    ❌ PREPROCESS FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: load_model
print("\n[4] Running load_model()...")
try:
    model = load_model()
    print(f"    ✅ Model loaded: {type(model)}")
except Exception as e:
    print(f"    ❌ LOAD_MODEL FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: predict (TIMED — this is what the judge measures)
print("\n[5] Running predict(df_processed, model)...")
try:
    from time import perf_counter
    t0 = perf_counter()
    predictions = predict(df_processed, model)
    elapsed = perf_counter() - t0
    print(f"    ✅ predict() completed in {elapsed:.4f}s")
    print(f"    Shape: {predictions.shape}")
    print(f"    Columns: {list(predictions.columns)}")
    print(f"    Dtypes: {dict(predictions.dtypes)}")
    print(f"    Sample:\n{predictions.head(3)}")
    
    # Validate output
    assert list(predictions.columns) == ['User_ID', 'Purchased_Coverage_Bundle'], \
        f"❌ Wrong columns: {list(predictions.columns)}"
    assert str(predictions['Purchased_Coverage_Bundle'].dtype) in ['int32','int64'], \
        f"❌ Wrong dtype: {predictions['Purchased_Coverage_Bundle'].dtype}"
    vals = set(predictions['Purchased_Coverage_Bundle'].unique())
    assert all(0 <= v <= 9 for v in vals), f"❌ Values out of range 0-9: {vals}"
    print(f"\n✅ All output validations PASSED")
    print(f"✅ VERDICT: GO — No container crash expected")
    
except Exception as e:
    print(f"    ❌ PREDICT FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

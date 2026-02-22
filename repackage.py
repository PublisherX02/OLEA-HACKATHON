"""Repackage submission.zip with the current solution.py (no retraining needed)."""
import zipfile

with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("solution.py")
    z.write("model.pkl")
    z.write("submission_requirements.txt", arcname="requirements.txt")

print("✅ submission.zip repackaged successfully!")
print()
print("Verifying fix...")
with zipfile.ZipFile("submission.zip") as z:
    for f in z.namelist():
        info = z.getinfo(f)
        print(f"  {f:<35} {info.file_size/1024:.1f} KB")
    
    sol_code = z.read("solution.py").decode()
    if "df.drop(columns" in sol_code and "user_ids" in sol_code:
        print()
        print("✅ predict() correctly extracts User_ID then drops it before model.predict()")
    else:
        print()
        print("❌ FIX NOT APPLIED - check solution.py!")

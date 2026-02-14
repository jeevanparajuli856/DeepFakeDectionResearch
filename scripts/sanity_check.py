import pandas as pd

df = pd.read_csv("data/manifests/master_manifest_new.csv")
bad = df[~df["path"].str.startswith("data/images/")]

if len(bad) > 0:
    print("❌ Invalid paths detected:")
    print(bad[["image_id", "path"]].head())
else:
    print("✅ All paths are valid.")

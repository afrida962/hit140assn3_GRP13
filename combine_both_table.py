import pandas as pd
import os

clean_dir = os.path.join(os.getcwd(), "cleaned_data")
perf_A = pd.read_csv(os.path.join(clean_dir, "performance_investigationA.csv"))
perf_B = pd.read_csv(os.path.join(clean_dir, "performance_investigationB.csv"))

combined_perf = pd.concat([perf_B, perf_A], ignore_index=True)
print("\n=== Combined Model Performance ===")
print(combined_perf)

# Save for report
combined_perf.to_csv(os.path.join(clean_dir, "performance_summary_all.csv"), index=False)

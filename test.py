import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dementia_misinformation_dataset_500.csv")

# Split into features (X) and labels (y)
X = df["claim"]
y = df["label"]

# Perform 80/20 split with stratification (to balance classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert back to DataFrame
train_df = pd.DataFrame({"claim": X_train, "label": y_train})
test_df = pd.DataFrame({"claim": X_test, "label": y_test})

# Save CSVs
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

print("✅ Dataset split complete!")
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("adult.csv")  # Replace with actual file

# Label encode categorical columns
categorical_columns = ['occupation', 'education', 'marital-status', 'race', 'gender']
le_dict = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Define features and target
X = df[categorical_columns]
y = df['income']  # Should be encoded already as 0/1

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model properly 🔒
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the label encoders too
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(le_dict, f)

print("✅ model.pkl and label_encoders.pkl saved successfully")

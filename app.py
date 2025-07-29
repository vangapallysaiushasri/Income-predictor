import streamlit as st
import pandas as pd
import pickle

# 🔄 Load model and label encoders (replace with your actual paths)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

# 📊 Load dataset (for dropdown values)
df = pd.read_csv("adult.csv")  # Make sure this matches your data file

# 💬 App Title
st.title("Income Prediction App 💰")
st.write("Choose your details and see if the model predicts earning over $50K!")

# 📥 User Input Section
st.write("### 🧪 Try Your Own Prediction")

occ_input = st.selectbox("Occupation", sorted(df['occupation'].dropna().unique()))
edu_input = st.selectbox("Education", sorted(df['education'].dropna().unique()))
marital_input = st.selectbox("Marital Status", sorted(df['marital-status'].dropna().unique()))
race_input = st.selectbox("Race", sorted(df['race'].dropna().unique()))
gender_input = st.selectbox("Gender", sorted(df['gender'].dropna().unique()))

# 🔄 Encode user input using label encoders
occ_encoded = le_dict['occupation'].transform([occ_input])[0]
edu_encoded = le_dict['education'].transform([edu_input])[0]
marital_encoded = le_dict['marital-status'].transform([marital_input])[0]
race_encoded = le_dict['race'].transform([race_input])[0]
gender_encoded = le_dict['gender'].transform([gender_input])[0]

# 🧠 Prediction
user_data = [[occ_encoded, edu_encoded, marital_encoded, race_encoded, gender_encoded]]
prediction = model.predict(user_data)[0]

# 📝 Output
result = ">50K" if prediction == 1 else "<=50K"
st.success(f"💼 Predicted Income Bracket: **{result}**")
# 🧹 Function to process inputs before prediction
def preprocess_input(input_data):
    # In your case, inputs are already encoded, so you can return them directly
    return input_data
# ✅ Add this after inputs and model setup
if st.button('Predict'):
    input_data = [occ_encoded, edu_encoded, marital_encoded, race_encoded, gender_encoded]  # use correct variables
    processed_data = preprocess_input(input_data)  # Optional preprocessing step
    prediction = model.predict([processed_data])[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"💼 Predicted Income Bracket: **{result}**")

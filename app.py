import streamlit as st
import pickle
import pandas as pd
# Load model and scaler
model = pickle.load(open("logistic.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# UI
st.title("Product Recommendation System")

rating = st.slider("Enter Rating", 1, 5)
helpful = st.number_input("Enter Helpful Votes", min_value=0)

if st.button("Predict"):
    
    if (rating<=2):
        if (helpful==0):
            st.success("✅ Product is Recommended")
        else:
            st.error("❌ Product is Not Recommended")

    else:
        input_df = pd.DataFrame([[rating, helpful]], 
                            columns=['reviews.rating', 'reviews.numHelpful'])
        input_data = scaler.transform(input_df)
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("✅ Product is Recommended")
        else:
            st.error("❌ Product is Not Recommended")

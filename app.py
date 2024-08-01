import streamlit as st
import pandas as pd
import pickle

# Load the model
model_path = 'best_model.pkl'
try:
    with open(model_path, 'rb') as file:
        best_model = pickle.load(file)
except FileNotFoundError:
    st.error('Model file not found. Ensure the model is trained and saved correctly.')

# Streamlit application
def main():
    st.title('Water Quality Prediction')

    # Collect user input
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, step=0.01)
    hardness = st.number_input('Hardness', min_value=0.0, step=0.01)
    solids = st.number_input('Solids', min_value=0.0, step=0.01)
    chloramines = st.number_input('Chloramines', min_value=0.0, step=0.01)
    sulfate = st.number_input('Sulfate', min_value=0.0, step=0.01)
    conductivity = st.number_input('Conductivity', min_value=0.0, step=0.01)
    organic_carbon = st.number_input('Organic Carbon', min_value=0.0, step=0.01)
    trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, step=0.01)
    turbidity = st.number_input('Turbidity', min_value=0.0, step=0.01)

    input_data = pd.DataFrame({
        'ph': [ph],
        'Hardness': [hardness],
        'Solids': [solids],
        'Chloramines': [chloramines],
        'Sulfate': [sulfate],
        'Conductivity': [conductivity],
        'Organic_carbon': [organic_carbon],
        'Trihalomethanes': [trihalomethanes],
        'Turbidity': [turbidity]
    })

    # Check for missing values and fill with mean (or any suitable strategy)
    input_data = input_data.fillna(input_data.mean())

    # Make prediction
    try:
        prediction = best_model.predict(input_data)[0]
        prediction_proba = best_model.predict_proba(input_data)[0]
        if prediction == 1:
            st.success('The water is potable.')
        else:
            st.error('The water is not potable.')
        st.write(f'Prediction probability: {prediction_proba}')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')

if __name__ == '__main__':
    main()


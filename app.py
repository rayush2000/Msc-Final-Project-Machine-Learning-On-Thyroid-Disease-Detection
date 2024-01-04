import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import warnings
import urllib.parse

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the machine learning model
try:
    pickled_model = pickle.load(open('random_forest_model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    pickled_model = None

# Create a Flask web application
app = Flask(__name__)

# Function to convert a value to float with error handling
def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        if isinstance(value, str) and value.strip() == '':
            return 0.0
        raise

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form data
        age = float(request.form.get('age', False))
        sex = float(request.form.get('sex', False))
        TSH = float(request.form.get('TSH', False))
        T3 = float(request.form.get('T3', False))
        T4U = float(request.form.get('T4U', False))
        FTI = float(request.form.get('FTI', False))
        onthyroxine = float(request.form.get('onthyroxine', False))
        queryonthyroxine = float(request.form.get('queryonthyroxine', False))
        onantithyroidmedication = float(request.form.get('onantithyroidmedication', False))
        sick = float(request.form.get('sick', False))
        pregnant = float(request.form.get('pregnant', False))
        thyroidsurgery = float(request.form.get('thyroidsurgery', False))
        I131treatment = float(request.form.get('I131treatment', False))
        queryhypothyroid = float(request.form.get('queryhypothyroid', False))
        queryhyperthyroid = float(request.form.get('queryhyperthyroid', False))
        lithium = float(request.form.get('lithium', False))
        goitre = float(request.form.get('goitre', False))
        tumor = float(request.form.get('tumor', False))
        hypopituitary = float(request.form.get('hypopituitary', False))
        psych = float(request.form.get('psych', False))

        # Create a dictionary with extracted values
        values = {
        "age": age, "sex": sex, "TSH": TSH, "T3": T3, "T4U": T4U, "FTI": FTI,
        "onthyroxine": onthyroxine, "queryonthyroxine": queryonthyroxine,
        "onantithyroidmedication": onantithyroidmedication,
        "sick": sick, "pregnant": pregnant, "thyroidsurgery": thyroidsurgery,
        "I131treatment": I131treatment,
        "queryhypothyroid": queryhypothyroid, "queryhyperthyroid": queryhyperthyroid,
        "lithium": lithium, "goitre": goitre, "tumor": tumor,
        "hypopituitary": hypopituitary,
        "psych": psych
    }

        # Insert data into MongoDB
        # insert_data = db.insert_one(values)

        # Create a DataFrame from the dictionary
        df_transform = pd.DataFrame.from_dict([values])

        # Perform feature transformations
        df_transform.age = df_transform['age'] ** (1 / 2)
        df_transform.TSH = np.log1p(df_transform['TSH'])
        df_transform.T3 = df_transform['T3'] ** (1 / 2)
        df_transform.T4U = np.log1p(df_transform['T4U'])
        df_transform.FTI = df_transform['FTI'] ** (1 / 2)

        # Convert DataFrame to dictionary
        values = df_transform.to_dict(orient='records')[0]

        # Create a NumPy array for model prediction
        arr = np.array([[values['age'], values['sex'], values['TSH'], values['T3'], values['T4U'], values['FTI'],
                         values['onthyroxine'], values['queryonthyroxine'], values['onantithyroidmedication'],
                         values['sick'], values['pregnant'], values['thyroidsurgery'], values['I131treatment'],
                         values['queryhypothyroid'], values['queryhyperthyroid'], values['lithium'], values['goitre'],
                         values['tumor'], values['hypopituitary'], values['psych']]])

        # Make predictions using the model
        pred = pickled_model.predict(arr)[0]

        # Determine the result based on the prediction
        if pred == 0:
            res_Val = "Hyperthyroid"
        elif pred == 1:
            res_Val = "Hypothyroid"
        else:
            res_Val = 'Negative'

        # Render the result template with the prediction
        return render_template('result.html', prediction_text='Result: {}'.format(res_Val))
    except Exception as e:
        return f"An error occurred: {e}"

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5000)

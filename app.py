from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
gbc_model = joblib.load('/content/gradient_boosting_classifier_model.joblib')

# Load the LabelEncoder
le = joblib.load('/content/label_encoder.joblib')

# Assuming X from previous steps is in scope for column order, or manually define it.
# For a standalone API, X's columns would need to be stored or explicitly defined.
# For this notebook context, we can assume X is available or reconstruct its column names.
# X = df_processed.drop(['Dosha', 'Dosha_encoded'], axis=1) # This line was executed previously.
# Let's ensure X's columns are available if not in the global scope.
# If running as a standalone script, one would typically load X_columns from a file
# or define them manually based on the training data.

# For the purpose of this notebook, we can recreate X's columns or assume X is in scope
# Let's assume the previous df_processed was used to get X, and we need its columns.
# This is a bit tricky in an interactive environment without global scope guarantees after previous cells execute.
# To be robust, one might save X.columns as a list during training.

# As a workaround for this interactive session, we'll try to get X.columns directly if 'X' is available.
# If 'X' is not available, then this would be a point where a standalone script would need
# to have a predefined list of feature names.

# Assuming X is available in the current kernel's scope from previous cells
# If this were a standalone script, we'd need to explicitly load or define these column names.

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        try:
            input_df = pd.DataFrame([data])

            # Ensure the order of columns matches the training data (X.columns)
            # Access the global X dataframe for column order
            global X 
            if 'X' not in globals():
                # Fallback if X is not found (e.g., in a fresh kernel session)
                # This list should ideally be loaded from a saved file along with the model and encoder
                feature_columns = ['rough_percent', 'whitish_percent', 'pale_dry_percent',
                                   'aspect_ratio_eye', 'dominant_hue', 'redness_eye', 'white_area',
                                   'FAR', 'JawAngle', 'ChinSharpnessAngle', 'Aspect_Ratio',
                                   'Nostril_Asymmetry', 'Bent_Angle']
                input_df = input_df[feature_columns]
            else:
                input_df = input_df[X.columns]

            # Make prediction
            prediction_encoded = gbc_model.predict(input_df)

            # Decode the prediction back to original Dosha name
            prediction_dosha = le.inverse_transform(prediction_encoded)

            return jsonify({'Dosha_prediction': prediction_dosha[0]})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

print("'/predict' endpoint defined.")

if __name__ == '__main__':
    print("Starting Flask app...")
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production

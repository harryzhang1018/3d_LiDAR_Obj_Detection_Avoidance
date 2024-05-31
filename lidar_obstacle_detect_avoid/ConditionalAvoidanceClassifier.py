import joblib

def load_classifier(filename='classifier_model.pkl'):
    # Load the classifier and scaler from a file
    model = joblib.load(filename)
    classifier = model['classifier']
    scaler = model['scaler']
    return classifier, scaler

def predict_labels(classifier, scaler, new_features):
    # Standardize the new features
    new_features_scaled = scaler.transform(new_features)
    # Predict labels for the new features
    predictions = classifier.predict(new_features_scaled)
    return predictions
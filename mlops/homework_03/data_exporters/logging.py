import mlflow
import os
import pickle


# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_experiment("homework_03")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):

    dv, model = data

    mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

    # Ensure the directory exists
    os.makedirs('models', exist_ok=True)
    with open("models/vectorizer.pkl","wb") as f_out:
        pickle.dump(dv, f_out)
    
    mlflow.log_artifact("models/vectorizer.pkl", artifact_path="dictvectorizer")


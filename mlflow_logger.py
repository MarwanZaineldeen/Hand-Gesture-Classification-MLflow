import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd
import os
import tempfile
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report

def setup_mlflow(experiment_name="Hand_Gesture_Classification", db_name="mlflow_research.db"):
    tracking_uri = f"sqlite:///{db_name}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI set to: {tracking_uri}")
    print(f"Active Experiment: {experiment_name}")

def log_experiment(run_name, model, X_train, X_test, y_test, params, metrics, tags=None, fig=None, fig_name="artifact.png"):
    with mlflow.start_run(run_name=run_name):
        dataset = mlflow.data.from_pandas(X_train, name="Hand_Gesture_Landmarks")
        mlflow.log_input(dataset, context="training")
        
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        if tags:
            mlflow.set_tags(tags)
            
        predictions = model.predict(X_test)
        signature = infer_signature(X_test, predictions)
        
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            signature=signature
        )
        
        report = classification_report(y_test, predictions, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "per_class_metrics.csv")
            df_report.to_csv(csv_path)
            mlflow.log_artifact(csv_path)
        
        if fig:
            mlflow.log_figure(fig, fig_name)
            
        print(f"Successfully logged: '{run_name}' with metrics, signature, dataset, and artifacts.")
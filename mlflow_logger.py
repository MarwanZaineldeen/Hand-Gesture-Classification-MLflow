import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def setup_mlflow(experiment_name="Hand_Gesture_Classification", db_name="mlflow_research.db"):
    """Safely initializes the local SQLite database and sets the experiment."""
    tracking_uri = f"sqlite:///{db_name}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow Tracking URI set to: {tracking_uri}")
    print(f"Active Experiment: {experiment_name}")

def log_experiment(run_name, model, X_train, params, metrics, tags=None, fig=None, fig_name="artifact.png"):
    """Professionally logs parameters, metrics, tags, the model, and visual artifacts."""
    with mlflow.start_run(run_name=run_name):
        
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        if tags:
            mlflow.set_tags(tags)
            
        predictions = model.predict(X_train)
        signature = infer_signature(X_train, predictions)
        
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            signature=signature
        )
        
        if fig:
            mlflow.log_figure(fig, fig_name)
            
        print(f"Successfully logged: '{run_name}' with metrics, signature, and artifacts.")
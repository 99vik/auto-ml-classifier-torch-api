import json
from src.nn.engine import engine

def train_model_logic(params, progress_queue):
    progress_queue.put(json.dumps({"status": "preparing"}))
    try:
        model, data_by_labels, labels, total_params = engine(
            file=params['file'],
            label_index=params['label_index'],
            iterations=params['iterations'],
            learning_rate=params['learning_rate'],
            activation_function=params['activation_function'],
            progress_queue=progress_queue,
            hidden_layers=params['hidden_layers'],
            normalization=params['normalization'],
            train_ratio=params['train_ratio'],
            dropout=params['dropout'],
            random_seed=params['random_seed']
        )
        progress_queue.put(json.dumps({"status": "complete", "model": model, "dataByLabels": data_by_labels, "labels": labels, "totalParams": total_params}))
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")
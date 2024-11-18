from flask import Flask, request, Response
from flask_cors import CORS
from scripts.engine import engine
from scripts.utils import turn_json_to_torch
from scripts.nn import Model
import queue
import json
app = Flask(__name__)
CORS(app)

progress_queue = queue.Queue()

@app.post("/api/train_model")
def train_model():
    file = request.files['file']
    label_index = int(request.form['label_index'])
    iterations = int(request.form['iterations'])
    learning_rate = float(request.form['learning_rate'])
    activation_function = request.form['activation_function']
    normalization = request.form['normalization'].lower() == 'true'
    train_ratio = float(request.form['train_test_split'])/100
    dropout = float(request.form['dropout'])/100
    random_seed = request.form.get('random_seed')
    hidden_layers_str = request.form.get('hidden_layers', '')

    if hidden_layers_str:
        hidden_layers = [int(num) for num in hidden_layers_str.split(',')]
    else:
        hidden_layers = []

    progress_queue.put(json.dumps({"status": "preparing"}))
    try:
        model, data_by_labels, labels, total_params = engine(file=file, 
                                                             label_index=label_index, 
                                                             iterations=iterations, 
                                                             learning_rate=learning_rate, 
                                                             activation_function=activation_function, 
                                                             progress_queue=progress_queue, 
                                                             hidden_layers=hidden_layers, 
                                                             normalization=normalization, 
                                                             train_ratio=train_ratio, 
                                                             dropout=dropout,
                                                             random_seed= int(random_seed) if random_seed else None
                                                             )
    except Exception as e:
        progress_queue.put(json.dumps({"status": "error", "error": str(e)}))
        return Response('success', status=200)
        
    progress_queue.put(json.dumps({"status": "complete", "model": model, "dataByLabels": data_by_labels, "labels": labels, "totalParams": total_params}))

    return Response('success', status=200)

@app.route("/api/train_progress", methods=['GET'])
def train_progress():
    def generate():
        while True:
            try:
                progress = progress_queue.get()
                yield f"data: {progress}\n\n"
                if '"status": "complete"' in progress:
                    break
            except queue.Empty:
                yield f"data: {'status': 'waiting'}...\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.post("/api/predict")
def predict():
    import torch
    model_raw = request.json.get('model')
    input_size = request.json.get('inputSize')
    output_size = request.json.get('outputSize')
    activation_function = request.json.get('activationFunction')
    hidden_layers = request.json.get('hiddenLayers')    
    inputsRaw = request.json.get('inputs')
    normalization = request.json.get('normalization')

    model = Model(input_size=input_size, output_size=output_size, activation_function=activation_function, hidden_layers=hidden_layers, normalization=normalization, dropout=0.0)
    state_dict = turn_json_to_torch(model_raw)
    model.load_state_dict(state_dict)
    input = torch.tensor(inputsRaw).float()
    result = model(input).argmax(0).item()
    return Response(json.dumps({'prediction': result}), status=200)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000 , debug=True)
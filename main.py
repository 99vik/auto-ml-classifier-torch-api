from flask import Flask, request, Response
from flask_cors import CORS
from scripts.engine import engine
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
    hidden_layers_str = request.form.get('hidden_layers', '')

    if hidden_layers_str:
        hidden_layers = [int(num) for num in hidden_layers_str.split(',')]
    else:
        hidden_layers = []

    progress_queue.put(json.dumps({"status": "preparing"}))
    engine(file=file, label_index=label_index, iterations=iterations, learning_rate=learning_rate, activation_function=activation_function, progress_queue=progress_queue, hidden_layers=hidden_layers)
    progress_queue.put(json.dumps({"status": "complete"}))

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

if __name__ == "__main__":
    app.run(debug=True)
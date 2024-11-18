from flask import Flask, request, Response
from flask_cors import CORS
from src.utils import parse_train_request_params, parse_predict_request_params
from src.train_model_logic import train_model_logic
from src.predict_logic import predict_logic
import queue
import json
app = Flask(__name__)
CORS(app)

progress_queue = queue.Queue()

@app.post("/api/train_model")
def train_model():
    try:
        params = parse_train_request_params(request)
        train_model_logic(params, progress_queue)
    except Exception as e:
        progress_queue.put(json.dumps({"status": "error", "error": str(e)}))
        return Response('error', status=500)
        
    return Response('success', status=200)

@app.get("/api/train_progress")
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
    params = parse_predict_request_params(request)
    result = predict_logic(params)
    return Response(json.dumps({'prediction': result}), status=200)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000 , debug=True)
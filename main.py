from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from scripts.engine import engine
import time
import queue

app = Flask(__name__)
CORS(app)

progress_queue = queue.Queue()

@app.post("/api/train_model")
def train_model():
    file = request.files['file']
    label_index = int(request.form['label_index'])
    import threading
    threading.Thread(target=run_training, args=(file, label_index)).start()

    return jsonify({'message': 'training started'})

def run_training(file, label_index):
    engine(file, label_index, progress_queue)
    
    progress_queue.put("Training complete")

@app.route("/api/train_progress", methods=['GET'])
def train_progress():
    def generate():
        while True:
            try:
                progress = progress_queue.get()
                yield f"data: {progress}\n\n"
                if progress == "Training complete":
                    break
            except queue.Empty:
                yield f"data: Waiting for updates...\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)
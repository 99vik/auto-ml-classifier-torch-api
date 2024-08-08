from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.engine import engine

app = Flask(__name__)
CORS(app)

@app.post("/api/test")
def test_api():
    file = request.files['file']
    engine(file)

    return jsonify({'file_name': file.filename}) 

if __name__ == "__main__":
    app.run(debug=True)
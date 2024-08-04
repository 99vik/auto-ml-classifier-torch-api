from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
app = Flask(__name__)
CORS(app)

@app.post("/api/test")
def test_api():
    csv_file = request.files['file']
    file_contents = csv_file.read().decode('utf-8')
    print(file_contents)
    reader = csv.reader(file_contents.splitlines(), delimiter=',')
    for row in reader:
        print(row)

    return jsonify({'file_name': csv_file.filename}) 

if __name__ == "__main__":
    app.run(debug=True)
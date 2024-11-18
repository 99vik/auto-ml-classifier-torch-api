def parse_train_request_params(request):
    file = request.files['file']
    label_index = int(request.form['label_index'])
    iterations = int(request.form['iterations'])
    learning_rate = float(request.form['learning_rate'])
    activation_function = request.form['activation_function']
    normalization = request.form['normalization'].lower() == 'true'
    train_ratio = float(request.form['train_test_split']) / 100
    dropout = float(request.form['dropout']) / 100
    random_seed = request.form.get('random_seed')
    hidden_layers_str = request.form.get('hidden_layers', '')

    hidden_layers = [int(num) for num in hidden_layers_str.split(',')] if hidden_layers_str else []

    return {
        "file": file,
        "label_index": label_index,
        "iterations": iterations,
        "learning_rate": learning_rate,
        "activation_function": activation_function,
        "normalization": normalization,
        "train_ratio": train_ratio,
        "dropout": dropout,
        "random_seed": int(random_seed) if random_seed != 'false' else None,
        "hidden_layers": hidden_layers
    }

def parse_predict_request_params(request):
    import torch
    from src.nn.Model import Model
    from src.nn.helpers import turn_json_to_torch

    model_raw = request.json.get('model')
    input_size = request.json.get('inputSize')
    output_size = request.json.get('outputSize')
    activation_function = request.json.get('activationFunction')
    hidden_layers = request.json.get('hiddenLayers')    
    inputsRaw = request.json.get('inputs')
    normalization = request.json.get('normalization')

    return {
        "model_raw": model_raw,
        "input_size": input_size,
        "output_size": output_size,
        "activation_function": activation_function,
        "hidden_layers": hidden_layers,
        "inputsRaw": inputsRaw,
        "normalization": normalization
    }
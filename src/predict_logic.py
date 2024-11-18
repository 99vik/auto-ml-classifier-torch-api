import torch
from src.nn.Model import Model
from src.nn.helpers import turn_json_to_torch

def predict_logic(params):

    model = Model(input_size=params['input_size'], output_size=params['output_size'], activation_function=params['activation_function'], hidden_layers=params['hidden_layers'], normalization=params['normalization'], dropout=0.0)
    state_dict = turn_json_to_torch(params['model_raw'])
    model.load_state_dict(state_dict)
    input = torch.tensor(params['inputsRaw']).float()
    result = model(input).argmax(0).item()
    return result
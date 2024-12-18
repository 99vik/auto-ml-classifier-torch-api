import torch
from torch import nn
from src.nn.Model import Model
from src.nn.train_loop import train_loop
from src.nn.helpers import to_split_tensor_data, csv_parser
import json

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super(TensorEncoder, self).default(obj)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def engine(file, 
           label_index, 
           iterations, 
           learning_rate, 
           activation_function, 
           progress_queue, 
           hidden_layers, 
           normalization, 
           train_ratio, 
           dropout, 
           random_seed):
    try:
        labels, data, data_by_labels = csv_parser(label_index=label_index, file=file)
        X_tr, X_te, y_tr, y_te = to_split_tensor_data(labels, data, label_index, train_ratio, random_seed)
        X_tr = X_tr.to(device)
        X_te = X_te.to(device)
        y_tr = y_tr.to(device)
        y_te = y_te.to(device)

        if (random_seed):
            torch.manual_seed(random_seed)
        
        model = Model(input_size=len(X_tr[0]), 
                      output_size=len(labels), 
                      activation_function=activation_function, 
                      hidden_layers=hidden_layers, 
                      normalization=normalization, 
                      dropout=dropout).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loop(model, X_tr, y_tr, X_te, y_te, loss_fn, optimizer, iterations, progress_queue, labels, device)
        total_params = sum(p.numel() for p in model.parameters())

        state_dict = model.state_dict()
        json_state_dict = json.dumps(state_dict, cls=TensorEncoder)
    except Exception as e:
        raise RuntimeError(f'Error: {e}')
    
    return json_state_dict, data_by_labels, list(map(float, labels)), total_params
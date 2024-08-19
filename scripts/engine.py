import torch
from torch import nn
from scripts.csv_parser import csv_parser
from scripts.nn import Model, train_loop
from scripts.utils import to_split_tensor_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def engine(file, label_index, progress_queue):

    labels, data = csv_parser(label_index=label_index, file=file)

    X_tr, X_te, y_tr, y_te = to_split_tensor_data(labels, data, label_index)

    X_tr = X_tr.to(device)
    X_te = X_te.to(device)
    y_tr = y_tr.to(device)
    y_te = y_te.to(device)
        
    model = Model(input_size=len(X_tr[0]), output_size=len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    iterations = 200

    train_loop(model, X_tr, y_tr, X_te, y_te, loss_fn, optimizer, iterations, progress_queue)

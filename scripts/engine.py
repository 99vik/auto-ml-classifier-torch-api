import torch
from torch import nn
from scripts.csv_parser import csv_parser
from scripts.nn import Model, train_loop
from scripts.utils import to_split_tensor_data
from torchmetrics import Accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def engine(file, label_index, iterations, learning_rate, activation_function, progress_queue, hidden_layers):

    labels, data = csv_parser(label_index=label_index, file=file)

    X_tr, X_te, y_tr, y_te = to_split_tensor_data(labels, data, label_index)

    X_tr = X_tr.to(device)
    X_te = X_te.to(device)
    y_tr = y_tr.to(device)
    y_te = y_te.to(device)
        
    model = Model(input_size=len(X_tr[0]), output_size=len(labels), activation_function=activation_function, hidden_layers=hidden_layers).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loop(model, X_tr, y_tr, X_te, y_te, loss_fn, optimizer, iterations, progress_queue, labels, device)

    
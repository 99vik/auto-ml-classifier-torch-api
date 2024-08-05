import torch
from torch import nn
from csv_parser import csv_parser
from sklearn.model_selection import train_test_split
from utils import accuracy_fn
from nn import Model, train_loop

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    LABEL_INDEX = 1
    file = 'SVMtrain.csv'

    labels, data = csv_parser(label_index=LABEL_INDEX, file=file)

    y = []
    X = []
    for line in data:
        y.append(labels.index(line[LABEL_INDEX]))
        X.append([float(s) for s in (line[:LABEL_INDEX]+line[LABEL_INDEX+1:])]) 

    y = torch.tensor(y)
    X = torch.tensor(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

    X_tr = X_tr.to(device)
    X_te = X_te.to(device)
    y_tr = y_tr.to(device)
    y_te = y_te.to(device)
        
    model = Model(input_size=len(X_tr[0]), output_size=len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    iterations = 200

    train_loop(model, X_tr, y_tr, X_te, y_te, loss_fn, optimizer, iterations)

main()
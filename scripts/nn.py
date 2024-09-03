import torch
from torch import nn
import json
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

class Model(nn.Module):
        def __init__(self, input_size, output_size, activation_function, hidden_layers, normalization, dropout):
            super().__init__()
            self.activation = self.get_activation(activation_function)
            self.layer_stack = self.create_layers(input_size, output_size, hidden_layers, normalization, dropout)

        def get_activation(self, activation_function):
            if activation_function == 'relu':
                return nn.ReLU()
            elif activation_function == 'sigmoid':
                return nn.Sigmoid()
            elif activation_function == 'tanh':
                return nn.Tanh()
            elif activation_function == 'linear':
                return nn.Identity()
            
        def create_layers(self, input_size, output_size, hidden_layers, normalization, dropout):
            layers = []
            prev_size = input_size
            for size in hidden_layers:
                layers.append(nn.Linear(in_features=prev_size, out_features=size))
                layers.append(self.activation)
                if normalization:
                    print('Normalization: ', normalization)
                    layers.append(nn.LayerNorm(size))
                layers.append(nn.Dropout(p=dropout))
                prev_size = size
            layers.append(nn.Linear(in_features=prev_size, out_features=output_size))
            return nn.Sequential(*layers)

        def forward(self, x):
            return self.layer_stack(x)
        
def train_loop(model, X_tr, y_tr, X_te, y_te, loss_fn, optimizer, iterations, progress_queue, labels, device):
    accuracy = Accuracy(task='multiclass', num_classes=len(labels)).to(device)
    f1score = F1Score(task="multiclass", num_classes=len(labels)).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=len(labels)).to(device)

    for i in range(iterations + 1):
        model.train()
        logits = model(X_tr)
        logits_pred = torch.softmax(logits, dim=1).argmax(dim=1)
        acc = accuracy(logits_pred, y_tr).item()*100

        loss = loss_fn(logits, y_tr)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        model.eval()
        if (i) % (iterations / 20) == 0:
            with torch.inference_mode():
                test_logits = model(X_te)
                test_logits_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                test_loss = loss_fn(test_logits, y_te)
                test_acc = accuracy(test_logits_pred, y_te).item()*100
                f1 = f1score(test_logits_pred, y_te)
                confusion_matrix = confmat(test_logits_pred, y_te)

            training_data = {
                 "status": "training",
                 "iteration": str(i),
                 "trainLoss": loss.item(),
                 "trainAccuracy": acc,
                 "testLoss": test_loss.item(),
                 "testAccuracy": test_acc,
                 "f1Score": f1.item(),
                 "confusionMatrix": confusion_matrix.tolist()
            }
            progress_queue.put(json.dumps(training_data))


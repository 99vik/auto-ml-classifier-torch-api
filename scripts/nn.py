import torch
from torch import nn
from scripts.utils import accuracy_fn
import json

class Model(nn.Module):
        def __init__(self, input_size, output_size, activation_function, hidden_layers):
            super().__init__()
            self.activation = self.get_activation(activation_function)
            self.layer_stack = self.create_layers(input_size, output_size, hidden_layers)

            # self.layer_stack = nn.Sequential(
            #     nn.Linear(in_features=input_size, out_features=10),
            #     self.activation,
            #     # nn.BatchNorm1d(10),
            #     # nn.Dropout(p=0.02),
            #     nn.Linear(in_features=10, out_features=output_size),
            # )

        def get_activation(self, activation_function):
            if activation_function == 'relu':
                return nn.ReLU()
            elif activation_function == 'sigmoid':
                return nn.Sigmoid()
            elif activation_function == 'tanh':
                return nn.Tanh()
            elif activation_function == 'linear':
                return nn.Identity()
            
        def create_layers(self, input_size, output_size, hidden_layers):
            layers = []
            prev_size = input_size
            for size in hidden_layers:
                layers.append(nn.Linear(in_features=prev_size, out_features=size))
                layers.append(self.activation)
                prev_size = size
            layers.append(nn.Linear(in_features=prev_size, out_features=output_size))
            return nn.Sequential(*layers)

        def forward(self, x):
            return self.layer_stack(x)
        
def train_loop(model, X_tr, y_tr, X_te, y_te, loss_fn, optimizer, iterations, progress_queue):
     for i in range(iterations + 1):
        model.train()

        logits = model(X_tr)
        logits_pred = torch.softmax(logits, dim=1).argmax(dim=1)

        acc = accuracy_fn(y_target=y_tr, y_pred=logits_pred)
        loss = loss_fn(logits, y_tr)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_te)
            test_logits_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss = loss_fn(test_logits, y_te)
            test_acc = accuracy_fn(y_target=y_te, y_pred=test_logits_pred)



        if (i) % (iterations / 10) == 0:
            training_data = {
                 "status": "training",
                 "iteration": str(i),
                 "trainLoss": loss.item(),
                 "trainAccuracy": acc,
                 "testLoss": test_loss.item(),
                 "testAccuracy": test_acc
            }
            progress_queue.put(json.dumps(training_data))


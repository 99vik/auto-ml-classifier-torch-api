import torch
from torch import nn
from utils import accuracy_fn
class Model(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=10),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Dropout(p=0.02),
                nn.Linear(in_features=10, out_features=output_size),
            )

        def forward(self, x):
            return self.layer_stack(x)
        
def train_loop(model, X_tr, y_tr, X_te, y_te, loss_fn, optimizer, iterations):
     for i in range(iterations):
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

        if (i) % 20 == 0:
            print(f'Iteration {i}: TRAIN LOSS: {loss:.5f} | TRAIN ACCURACY: {acc:.1f}% | TEST_LOSS: {test_loss:.5f} | TEST ACCURACY: {test_acc:.1f}%')

    # print(f'Iteration {i+1}: TRAIN LOSS: {loss:.5f} | TRAIN ACCURACY: {acc:.2f}% | TEST_LOSS: {test_loss:.5f} | TEST ACCURACY: {test_acc:.2f}%')
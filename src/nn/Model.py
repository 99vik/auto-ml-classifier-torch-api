from torch import nn

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
                    layers.append(nn.LayerNorm(size))
                layers.append(nn.Dropout(p=dropout))
                prev_size = size
            layers.append(nn.Linear(in_features=prev_size, out_features=output_size))
            return nn.Sequential(*layers)

        def forward(self, x):
            return self.layer_stack(x)
        
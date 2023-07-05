import torch
import torch.nn as nn


class MemNetFC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_layers = len(kwargs['hidden_layers'])
        self.layers = [kwargs['hidden_layers'][0]]+kwargs['hidden_layers']
        self.input_layer = nn.Linear(
            in_features=kwargs["input_size"],
            out_features=kwargs['hidden_layers'][0]
        )
        self.hidden_layers = nn.Sequential(*[nn.Linear(
            in_features=kwargs['hidden_layers'][i-1], 
            out_features=kwargs['hidden_layers'][i]
        ) for i in range(1, self.n_layers)])

        self.decoder_output_layer = nn.Linear(
            in_features=kwargs['hidden_layers'][-1],
            out_features=kwargs["output_size"]
        )

    
    def forward_transposed(self, code):
        activation = torch.relu(torch.matmul(code,
                                             self.decoder_output_layer.weight))
        for layer in self.hidden_layers[::-1]:
            activation = torch.relu(torch.matmul(activation,
                                                 layer.weight))
        predimg = torch.matmul(activation,
                                        self.input_layer.weight)
        return predimg
    
    def forward(self, x):
        activation = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            activation = torch.relu(layer(activation))
        predlabel = self.decoder_output_layer(activation)
        return predlabel
        

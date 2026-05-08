import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence

class AmmoniaRNN(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_size, 
                 num_layers,
                 nonlinearity, 
                 bidirectional,
                 cat_dims = None, embedding_dims = None):
        
        super(AmmoniaRNN, self).__init__()
        
        D = 1 + 1 * bidirectional
           
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings = cat_dim, embedding_dim = embed_dim)
            for cat_dim, embed_dim in zip(cat_dims, embedding_dims)
        ])
        
        input_size = input_size - len(cat_dims) + sum(embedding_dims)           
        
        self.rnn = nn.RNN(input_size, 
                          hidden_size, 
                          num_layers = num_layers,
                          nonlinearity = nonlinearity, 
                          bidirectional = bidirectional)    
        
        self.fc1 = nn.Linear(hidden_size * D, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, output_size)

        self.relu = nn.ReLU()

    
    def forward(self, list_x_train, list_y_train = None):

        list_x = []
        sequence_lengths = []

        for x in list_x_train:

            x_continuous = x[0]
            x_categoricals = x[1]
                
            x_embeds = [embed(x_cat) for embed, x_cat in zip(self.embeddings, x_categoricals)]
        
            x_final = torch.cat([x_continuous] + x_embeds, dim = -1)
        
            list_x.append (x_final)
        
            sequence_lengths.append(len(x_final))

        sorted_lengths, sorted_idx = torch.sort(torch.tensor(sequence_lengths), descending=True)

        sorted_x = [list_x[i] for i in sorted_idx]
        packed_x = pack_sequence(sorted_x, enforce_sorted=False)

        if list_y_train is not None :
            
            sorted_y = [list_y_train[i] for i in sorted_idx]
            packed_y = pack_sequence(sorted_y, enforce_sorted=False)

        
        h_packed, h_n = self.rnn(packed_x)

        h = h_packed[0]

        out = self.fc1(h)
        out_fc1_before_relu = torch.sum (torch.abs (out))

        out = self.relu(out)
        out_fc1_after_relu = torch.sum (torch.abs (out))
        
        out = self.fc2(out)
        out_fc2_before_relu = torch.sum (torch.abs (out))
        
        out = self.relu(out)
        out_fc2_after_relu = torch.sum (torch.abs (out))
                    
        out = self.fc3(out)
        
        
        if list_y_train is not None :
            
            return out, packed_y

        else :

            return out, out_fc1_before_relu, out_fc1_after_relu, out_fc2_before_relu, out_fc2_after_relu, h_n

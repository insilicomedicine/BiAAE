import torch
from torch import nn
from .tokenizer import encode, get_vocab_size

# FOR IMAGE DATASET

class MnistCNNEncoder(nn.Module):
    def __init__(self, out_dim, dropout2d=0.2, dropout1d=0.2, short_tail=False):
        super(MnistCNNEncoder, self).__init__()

        self.out_dim = out_dim
        self.short_tail = short_tail
        
        self.dropout2d = dropout2d
        self.dropout1d = dropout1d
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Dropout2d(dropout2d),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, 3, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
        )

        if self.short_tail:
            self.fc = nn.Linear(64, out_dim)
        else:
            self.fc = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout1d),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout1d),
            nn.Linear(128, self.out_dim ),
        )

    def forward(self, x):
        z = self.conv(x).reshape(x.shape[0], -1)
        return self.fc(z)

    
# FOR EXPRESSIONS
class ExprDiffEncoder(nn.Module):
    def __init__(self, out_dim):
        super(ExprDiffEncoder, self).__init__()
        
        self.out_dim = out_dim
        
        self.exp_fc = nn.Sequential(
            nn.Linear(978, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        
        self.final_fc = nn.Sequential(
            nn.Linear(128 + 1, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, batch):
        diff, dose = torch.split(batch, 978, -1)
        
        z_diff = self.exp_fc(diff)
        
        return self.final_fc(torch.cat([z_diff, torch.log(dose + 0.001)], dim=-1))
        

# WRAPPER TO COMBINE FEW DECODERS
class JointEncoder(nn.Module):
    def __init__(self, enc_x, enc_y, out_dim):
        super(JointEncoder, self).__init__()

        self.enc_x = enc_x
        self.enc_y = enc_y

        self.out_dim = out_dim

        self.fc = nn.Sequential(
            nn.Linear(enc_x.out_dim + enc_y.out_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, xy_pair):
        x, y = xy_pair
        z = torch.cat((self.enc_x(x), self.enc_y(y)), dim=-1)
        return self.fc(z)

# FOR SMILES
class RNNEncoder(nn.Module):
    def __init__(self, out_dim, hidden_size=128, num_layers=2,
                 bidirectional=False):
        super(RNNEncoder, self).__init__()

        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, out_dim))

    def forward(self, sm_list):
        """
        Maps smiles onto a latent space
        """

        tokens, lens = encode(sm_list)
        to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)

        outputs = self.rnn(self.embs(to_feed))[0]
        outputs = outputs[lens, torch.arange(len(lens))]

        return self.final_mlp(outputs)


# WRAPPER FOR FINE TUNING
class FinetunedEncoder(nn.Module):
    def __init__(self, fr_enc, out_dim):
        super().__init__()
        self.fr_enc = fr_enc
        
        self.out_dim = out_dim
        
        self.step_counter = 0
        
        self.new_fc = nn.Sequential(nn.Linear(fr_enc.out_dim, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, out_dim))
                               
        for p in self.fr_enc.parameters():
            p.requires_grad = False
                
        for p in self.fr_enc.final_mlp.parameters():
            p.requires_grad = True
            
        self.parameters = nn.ParameterList(self.new_fc.parameters())
        
    def forward(self, x):
        self.step_counter += 1
        
        return self.new_fc(self.fr_enc(x))
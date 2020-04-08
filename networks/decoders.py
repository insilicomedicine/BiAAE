import torch
from torch import nn
from .tokenizer import encode, get_vocab_size, decode
import torch.nn.functional as F

# FOR IMAGES
class MnistCNNDecoder(nn.Module):
    def __init__(self, in_dim):
        super(MnistCNNDecoder, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(32, 64, 5, stride=3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )

    def forward(self, z):
        z = self.fc(z)
        return self.conv(z.reshape(z.shape[0], 16, 2, 2))

    def get_log_prob(self, x, z):
        return -torch.nn.MSELoss()(self.forward(z), x)

    def sample(self, z):
        return self.forward(z)

    
# FOR EXPRESSIONS
class ExprDiffDecoder(nn.Module):
    def __init__(self, in_dim):
        super(ExprDiffDecoder, self).__init__()
        
        self.in_dim = in_dim
        
        self.start_fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64 + 1)
        )
        
        self.expr_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 978),
        )
        
    def forward(self, z):
        hidden = self.start_fc(z)
        
        h_diff, dose_log = torch.split(hidden, 64, dim=-1)

        return torch.cat((self.expr_fc(h_diff), torch.exp(dose_log)), dim=-1)
    
    def get_log_prob(self, x, z):
        diff_pred, dose_pred = torch.split(self.forward(z), 978, dim=-1)
        diff, dose= torch.split(x, 978, dim=-1)
        
        return -6 * nn.MSELoss()(diff_pred, diff) - nn.MSELoss()(
            torch.log(dose_pred + 0.001), torch.log(dose + 0.001))
    
    def sample(self, z):
        return self.forward(z)

    
# WRAPPER FOR CONDITIONAL DECODERS

class ConditionedDecoder(nn.Module):
    def __init__(self, dec, cond):
        super(ConditionedDecoder, self).__init__()
        self.dec = dec
        self.cond = cond

    def get_log_prob(self, x, z, cond):
        dec_inp = torch.cat((z, self.cond(cond)), dim=-1)
        return self.dec.get_log_prob(x, dec_inp)

    def sample(self, z, cond):
        dec_inp = torch.cat((z, self.cond(cond)), dim=-1)
        return self.dec.sample(dec_inp)
    
# FOR SMILES
class RNNDecoder(nn.Module):
    def __init__(self, in_dim, hidden_size=128, num_layers=2,
                 bidirectional=False):
        super(RNNDecoder, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size + in_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        self.lat_fc = nn.Linear(in_dim, hidden_size)
        self.out_fc = nn.Linear(hidden_size, get_vocab_size())

    def get_log_prob(self, x, z):
        """
        Maps smiles onto a latent space
        """

        tokens, lens = encode(x)
        x = tokens.transpose(1, 0).to(self.embs.weight.device)

        x_emb = self.embs(x)

        z_0 = z.unsqueeze(0).repeat(x_emb.shape[0], 1, 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)

        h_0 = self.lat_fc(z)
        h_0 = h_0.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        
        output, _ = self.rnn(x_input, h_0)
        y = self.out_fc(output)
         
        recon_loss = -F.cross_entropy(
            y.transpose(1, 0)[:, :-1].contiguous().view(-1, y.size(-1)),
            x.transpose(1, 0)[:, 1:].contiguous().view(-1),
            ignore_index=0,
            reduction='none'
        )
        
        recon_loss = (recon_loss.view(x.shape[1], -1).sum(dim=-1) / torch.tensor(lens).to(z_0.device).float())
        
        recon_loss = recon_loss.mean()
        
        return recon_loss
    
    def sample(self, z):
        with torch.no_grad():
            n_batch = z.shape[0]
            z_0 = z.unsqueeze(0)

            max_len = 100
            
            # Initial values
            h = self.lat_fc(z)
            h = h.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
            w = torch.tensor(1, device=z.device).repeat(n_batch)
            x = torch.tensor(0, device=z.device).repeat(n_batch, max_len)
            x[:, 0] = 1
            end_pads = torch.tensor([max_len], device=z.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                   device=z.device).bool()

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.embs(w).unsqueeze(0)
                
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.rnn(x_input, h)
                y = self.out_fc(o.squeeze(1))
                y = F.softmax(y, dim=-1)

                #w = torch.multinomial(y[0], 1)[:, 0]
                w = torch.max(y[0], dim=-1)[1]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == 2)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x += decode(x[i, 1:end_pads[i]].unsqueeze(0))

            return new_x
    
class FinetunedDecoder(nn.Module):
    def __init__(self, fr_dec, in_dim):
        super().__init__()
        
        self.fr_dec = fr_dec
        self.in_dim = in_dim
        
        self.new_fc = nn.Sequential(nn.Linear(in_dim, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, fr_dec.in_dim))
            
        self.step_counter = 0
 
        for p in self.fr_dec.parameters():
            p.requires_grad = False
                
        for p in self.fr_dec.lat_fc.parameters():
            p.requires_grad = True

        self.parameters = nn.ParameterList(self.new_fc.parameters())
        
    def get_log_prob(self, x, z):        
        return self.fr_dec.get_log_prob(x, self.new_fc(z))
    
    def sample(self, z):
        return self.fr_dec.sample(self.new_fc(z))

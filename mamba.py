import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from os import path
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, device):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        self.device = device

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, device=device)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            device=device
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, device=device)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, device=device)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, device=device)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # [batch, seq_len, 2 * d_inner]
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)

        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        
        # Compute A, D
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_inner)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        # u: (b, l, d_in)
        # delta: (b, l, d_in)
        # A: (d_in, n)
        # B: (b, l, n)
        # C: (b, l, n)
        # D: (d_in)
        
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize A and B
        # deltaA = exp(delta * A)
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bld,bln->bldn', delta, u, B)
        
        # Scan
        x = torch.zeros((b, d_in, n), device=self.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.sum(x * C[:, i].unsqueeze(1), dim=-1)
            ys.append(y)
            
        y = torch.stack(ys, dim=1) # (b, l, d_in)
        
        y = y + u * D
        return y

class Mamba(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, hidden_dim, num_hidden, maxLength, state_fname, device, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.state_fname = state_fname
        self.maxLength = maxLength
        self.num_embeddings = num_embeddings
        self.device = device
        
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx, device=self.device)
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=embedding_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                device=device
            ) for _ in range(num_hidden)
        ])
        
        self.norm_f = RMSNorm(embedding_dim, device=device)
        self.linear = nn.Linear(embedding_dim, num_embeddings, bias=False, device=device)

    def loadState(self):
        if path.isfile(self.state_fname):
            try:
                self.load_state_dict(torch.load(self.state_fname))
            except RuntimeError as err:
                utils.logger.warning(f"Skip loading checkpoint due to mismatch: {err}")
        else:
            utils.logger.info("State file is not found")

    def saveState(self):
        dir_name = path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)

    def forward(self, X):
        # X: [batch_size, seq_len]
        x = self.embedding(X.to(self.device))
        
        for layer in self.layers:
            x = layer(x) + x # Residual connection
            
        x = self.norm_f(x)
        output = self.linear(x)
        return output

    def sample(self, batch_size, temperature=1.0, eval=True):
        if eval:
            self.eval()
            nlls = None
        else:
            self.train()
            nlls = torch.zeros(
                (batch_size, 1), dtype=torch.float32, device=self.device)
        
        samples = torch.zeros((batch_size, self.maxLength), dtype=torch.long, device=self.device)
        input_seq = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        
        # Note: This naive sampling is inefficient for Mamba because it recomputes the whole sequence.
        # A proper implementation would use the recurrent state of the SSM.
        # For simplicity and consistency with the provided codebase structure, we use the naive approach here.
        
        for i in range(self.maxLength):
            x = self.embedding(input_seq)
            
            # Forward pass through layers
            for layer in self.layers:
                x = layer(x) + x
            
            x = self.norm_f(x)
            last_output = x[:, -1, :]
            y = self.linear(last_output)
            
            if temperature != 1.0:
                y *= temperature
            
            probs = F.softmax(y, dim=1)
            next_token = torch.multinomial(probs, 1)
            
            samples[:, i] = next_token.data[:, 0]
            input_seq = torch.cat([input_seq, next_token], dim=1)
            
            if self.training:
                 nlls = nlls + F.nll_loss(
                    torch.log(probs), samples[:, i].clone(), reduction='none').view((-1, 1))

        return samples.cpu(), nlls

    def loss_per_sample(self, pred_y, y):
        return F.cross_entropy(pred_y.to(self.device).transpose(1, 2), y.to(self.device), reduction='none').sum(dim=1).view((-1, 1))

    def trainModel(self, dataloader, optimizer, scheduler, nepoch, maxLength, tokenizer, printInterval, is_valid_fn=None):
        self.loadState()
        minloss = None
        numSample = 100
        if is_valid_fn is None:
            is_valid_fn = utils.isValidSmiles
        for epoch in range(1, nepoch + 1):
            lossList, accumulatedLoss, numValid_list = [], 0, []
            for nbatch, (X, y) in enumerate(dataloader, 1):
                self.train()
                pred_y = self(X)
                loss = self.loss_per_sample(pred_y, y).sum() / X.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossList.append(loss.item())
                accumulatedLoss += loss.item()
                if (nbatch == 1 or nbatch % printInterval == 0):
                    samples, _ = self.sample(100)
                    smilesStrs = tokenizer.getSmiles(samples)
                    numValid = sum([is_valid_fn(sm) for sm in smilesStrs])
                    numValid_list.append(numValid)
                    utils.logger.info(f'Epoch {epoch:3d} & Batch {nbatch:4d}: Loss= {sum(lossList) / len(lossList):.5e} Valid= {numValid:3d}/{numSample:3d}')
                    lossList.clear()
                    if minloss is None:
                        minloss = loss.item()
                    elif loss.item() < minloss:
                        self.saveState()
                        minloss = loss.item()
            scheduler.step()
            utils.logger.info(f'Epoch {epoch:3d}: Loss= {accumulatedLoss / nbatch:.5e} Valid= {round(sum(numValid_list) / len(numValid_list)):3d}/{numSample:3d}')

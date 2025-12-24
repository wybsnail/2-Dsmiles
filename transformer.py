import torch
import utils
from os import path
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class Transformer(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, hidden_dim, num_hidden, num_heads, maxLength, state_fname, device):
        super().__init__()
        self.state_fname = state_fname
        self.maxLength = maxLength
        self.num_embeddings = num_embeddings
        self.device = device
        
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, padding_idx, device=self.device)
        self.pos_encoder = PositionalEncoding(embedding_dim, maxLength)
        
        # Note: embedding_dim must be divisible by num_heads
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True,
            device=self.device
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=num_hidden)
        
        self.linear = torch.nn.Linear(
            embedding_dim, num_embeddings, device=self.device)

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

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, X):
        # X: [batch_size, seq_len]
        src = self.embedding(X.to(self.device))
        src = self.pos_encoder(src)
        mask = self._generate_square_subsequent_mask(X.size(1))
        output = self.transformer_encoder(src, mask=mask)
        output = self.linear(output)
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
        
        # Initial input: [batch_size, 1] (start tokens, assumed to be 0)
        input_seq = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        
        for i in range(self.maxLength):
            # Forward pass with current sequence
            src = self.embedding(input_seq)
            src = self.pos_encoder(src)
            mask = self._generate_square_subsequent_mask(input_seq.size(1))
            output = self.transformer_encoder(src, mask=mask)
            
            # Get last time step output
            last_output = output[:, -1, :] # [batch_size, embedding_dim]
            y = self.linear(last_output)   # [batch_size, num_embeddings]
            
            if temperature != 1.0:
                y *= temperature
            
            probs = torch.nn.functional.softmax(y, dim=1)
            next_token = torch.multinomial(probs, 1) # [batch_size, 1]
            
            # Store sample
            samples[:, i] = next_token.data[:, 0]
            
            # Update input_seq for next step
            input_seq = torch.cat([input_seq, next_token], dim=1)
            
            if self.training:
                 nlls = nlls + torch.nn.functional.nll_loss(
                    torch.log(probs), samples[:, i].clone(), reduction='none').view((-1, 1))

        return samples.cpu(), nlls

    def loss_per_sample(self, pred_y, y):
        return torch.nn.functional.cross_entropy(pred_y.to(self.device).transpose(1, 2), y.to(self.device), reduction='none').sum(dim=1).view((-1, 1))

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

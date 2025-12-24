import os
import torch
import utils


def _make_mlp(input_dim, hidden_dims, output_dim, device):
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1], device=device))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class AffineCoupling(torch.nn.Module):
    def __init__(self, dim, hidden_dims, mask, device) -> None:
        super().__init__()
        self.register_buffer('mask', mask)
        self.net = _make_mlp(dim, hidden_dims, dim * 2, device)

    def forward(self, x):
        x_masked = x * self.mask
        shift_scale = self.net(x_masked)
        shift, log_scale = shift_scale.chunk(2, dim=1)
        log_scale = torch.tanh(log_scale)
        y = x_masked + (1 - self.mask) * (x * torch.exp(log_scale) + shift)
        log_det = ((1 - self.mask) * log_scale).sum(dim=1)
        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask
        shift_scale = self.net(y_masked)
        shift, log_scale = shift_scale.chunk(2, dim=1)
        log_scale = torch.tanh(log_scale)
        x = y_masked + (1 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        log_det = -((1 - self.mask) * log_scale).sum(dim=1)
        return x, log_det


class Flow(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, hidden_dims, num_coupling, state_fname, device) -> None:
        super().__init__()
        self.device = device
        self.state_fname = state_fname
        self.maxLength = maxLength
        self.num_vocabs = num_vocabs
        self.dim = maxLength * num_vocabs
        masks = []
        for i in range(num_coupling):
            if i % 2 == 0:
                m = torch.cat([torch.ones(self.dim // 2), torch.zeros(self.dim - self.dim // 2)])
            else:
                m = torch.cat([torch.zeros(self.dim // 2), torch.ones(self.dim - self.dim // 2)])
            masks.append(m.to(torch.float32).to(self.device))
        self.couplings = torch.nn.ModuleList([
            AffineCoupling(self.dim, hidden_dims, mask, device) for mask in masks
        ])

    def loadState(self):
        if os.path.isfile(self.state_fname):
            try:
                self.load_state_dict(torch.load(self.state_fname))
            except RuntimeError as err:
                utils.logger.warning(f"Skip loading flow checkpoint due to mismatch: {err}")
        else:
            utils.logger.info('flow state file is not found')

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)

    def forward(self, X):
        z = X.reshape(X.shape[0], -1)
        total_log_det = torch.zeros(X.shape[0], device=self.device)
        for c in self.couplings:
            z, log_det = c(z)
            total_log_det += log_det
        return z, total_log_det

    def inverse(self, z):
        log_det_total = torch.zeros(z.shape[0], device=self.device)
        x = z
        for c in reversed(self.couplings):
            x, log_det = c.inverse(x)
            log_det_total += log_det
        return x, log_det_total

    def nll(self, X):
        z, log_det = self.forward(X)
        log_prob_z = -0.5 * (z ** 2).sum(dim=1)
        nll = -(log_prob_z + log_det)
        return nll

    def sample(self, nSample, temperature=1.0):
        z = torch.randn((nSample, self.dim), device=self.device) * temperature
        x, _ = self.inverse(z)
        x = x.view(nSample, self.maxLength, self.num_vocabs)
        probs = torch.nn.functional.softmax(x, dim=-1)
        token_ids = torch.argmax(probs, dim=-1) + 2
        return token_ids.cpu(), probs.cpu()

    def reconstruction_quality_per_sample(self, X):
        z, _ = self.forward(X)
        return -torch.norm(z, dim=1)

    def latent_space_quality(self, nSample, tokenizer, is_valid_fn=None):
        if is_valid_fn is None:
            is_valid_fn = utils.isValidSmiles
        numVectors, _ = self.sample(nSample)
        smilesStrs = tokenizer.getSmiles(numVectors)
        validSmilesStrs = [sm for sm in smilesStrs if is_valid_fn(sm)]
        return len(validSmilesStrs)

    def trainModel(self, dataloader, optimizer, scheduler, nepoch, tokenizer, printInterval, is_valid_fn=None):
        self.loadState()
        best_valid = 0
        numSample = 100
        if is_valid_fn is None:
            is_valid_fn = utils.isValidSmiles
        for epoch in range(1, nepoch + 1):
            loss_epoch = 0.0
            loss_interval = []
            quality_list, valid_list = [], []
            for nBatch, X in enumerate(dataloader, 1):
                X = X.to(self.device)
                optimizer.zero_grad()
                nll = self.nll(X)
                loss = nll.mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                loss_interval.append(loss.item())
                if nBatch == 1 or nBatch % printInterval == 0:
                    quality = self.reconstruction_quality_per_sample(X).mean()
                    numValid = self.latent_space_quality(numSample, tokenizer, is_valid_fn=is_valid_fn)
                    quality_list.append(quality)
                    valid_list.append(numValid)
                    utils.logger.info(
                        f"Epoch {epoch:4d} & Batch {nBatch:4d}: NLL= {sum(loss_interval) / len(loss_interval):.5e} Quality= {quality:.0f} Valid= {numValid:3d}/{numSample:3d}"
                    )
                    loss_interval.clear()
                    if numValid > best_valid:
                        self.saveState()
                        best_valid = numValid
            scheduler.step()
            mean_quality = sum(quality_list) / len(quality_list) if quality_list else 0.0
            mean_valid = sum(valid_list) / len(valid_list) if valid_list else 0.0
            utils.logger.info(
                f"Epoch {epoch:4d}: NLL= {loss_epoch / nBatch:.5e} Quality= {mean_quality:.0f} Valid= {mean_valid:.0f}/{numSample:3d}"
            )

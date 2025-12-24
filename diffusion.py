import math
import os
import torch
import utils


def sinusoidal_embedding(timesteps, dim, device):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / float(half)
    )
    angles = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((timesteps.shape[0], 1), device=device)], dim=1)
    return emb


def _make_mlp(input_dim, hidden_dims, output_dim, device):
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1], device=device))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class Denoiser(torch.nn.Module):
    def __init__(self, dim, hidden_dims, time_dim, device) -> None:
        super().__init__()
        self.time_proj = torch.nn.Linear(time_dim, time_dim, device=device)
        self.net = _make_mlp(dim + time_dim, hidden_dims, dim, device)

    def forward(self, x, t_emb):
        h = torch.cat([x, self.time_proj(t_emb)], dim=1)
        return self.net(h)


class Diffusion(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, hidden_dims, timesteps, beta_start, beta_end, state_fname, device) -> None:
        super().__init__()
        self.device = device
        self.state_fname = state_fname
        self.maxLength = maxLength
        self.num_vocabs = num_vocabs
        self.dim = maxLength * num_vocabs
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.denoiser = Denoiser(self.dim, hidden_dims, time_dim=hidden_dims[0], device=device)

    def loadState(self):
        if os.path.isfile(self.state_fname):
            try:
                self.load_state_dict(torch.load(self.state_fname))
            except RuntimeError as err:
                utils.logger.warning(f"Skip loading diffusion checkpoint due to mismatch: {err}")
        else:
            utils.logger.info('diffusion state file is not found')

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_losses(self, x_start, t, noise):
        x_noisy = self.q_sample(x_start, t, noise)
        t_emb = sinusoidal_embedding(t, self.denoiser.time_proj.in_features, self.device)
        noise_pred = self.denoiser(x_noisy, t_emb)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        t_emb = sinusoidal_embedding(t, self.denoiser.time_proj.in_features, self.device)
        noise_pred = self.denoiser(x, t_emb)
        beta_t = self.betas[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        mean = (1 / sqrt_alpha) * (x - beta_t / sqrt_one_minus * noise_pred)
        if t[0] == 0:
            return mean
        noise = torch.randn_like(x)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, nSample):
        x = torch.randn((nSample, self.dim), device=self.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((nSample,), step, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
        x = x.view(nSample, self.maxLength, self.num_vocabs)
        probs = torch.nn.functional.softmax(x, dim=-1)
        token_ids = torch.argmax(probs, dim=-1) + 2
        return token_ids.cpu(), probs.cpu()

    def latent_space_quality(self, nSample, tokenizer, is_valid_fn=None):
        if is_valid_fn is None:
            is_valid_fn = utils.isValidSmiles
        ids, _ = self.sample(nSample)
        smilesStrs = tokenizer.getSmiles(ids)
        validSmilesStrs = [sm for sm in smilesStrs if is_valid_fn(sm)]
        return len(validSmilesStrs)

    def trainModel(self, dataloader, optimizer, scheduler, nepoch, tokenizer, printInterval, is_valid_fn=None):
        self.loadState()
        best_valid = 0
        numSample = 50
        if is_valid_fn is None:
            is_valid_fn = utils.isValidSmiles
        for epoch in range(1, nepoch + 1):
            loss_epoch = 0.0
            loss_interval = []
            valid_list = []
            for nBatch, X in enumerate(dataloader, 1):
                X = X.to(self.device)
                x_flat = X.view(X.shape[0], -1)
                t = torch.randint(0, self.timesteps, (X.shape[0],), device=self.device)
                noise = torch.randn_like(x_flat)
                loss = self.p_losses(x_flat, t, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                loss_interval.append(loss.item())
                if nBatch == 1 or nBatch % printInterval == 0:
                    numValid = self.latent_space_quality(numSample, tokenizer, is_valid_fn=is_valid_fn)
                    valid_list.append(numValid)
                    utils.logger.info(
                        f"Epoch {epoch:4d} & Batch {nBatch:4d}: Diffusion_Loss= {sum(loss_interval) / len(loss_interval):.5e} Valid= {numValid:3d}/{numSample:3d}"
                    )
                    loss_interval.clear()
                    if numValid > best_valid:
                        self.saveState()
                        best_valid = numValid
            scheduler.step()
            mean_valid = sum(valid_list) / len(valid_list) if valid_list else 0.0
            utils.logger.info(
                f"Epoch {epoch:4d}: Diffusion_Loss= {loss_epoch / nBatch:.5e} Valid= {mean_valid:.0f}/{numSample:3d}"
            )

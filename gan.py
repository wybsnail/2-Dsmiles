import os
import torch
import utils


class Generator(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, latent_dim, hidden_dims, state_fname, device) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.device = device
        self.maxLength = maxLength
        self.num_vocabs = num_vocabs
        self.latent_dim = latent_dim
        output_dim = maxLength * num_vocabs
        dims = [latent_dim] + list(hidden_dims)
        if dims[-1] != output_dim:
            dims.append(output_dim)
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1], device=self.device))
            if idx < len(dims) - 2:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, noise, temperature=1.0):
        noise = noise.to(self.device)
        logits = self.net(noise)
        logits = logits.view(-1, self.maxLength, self.num_vocabs)
        if temperature != 1.0:
            logits = logits / max(temperature, 1.0e-6)
        return torch.nn.functional.softmax(logits, dim=-1)

    def loadState(self):
        if os.path.isfile(self.state_fname):
            try:
                self.load_state_dict(torch.load(self.state_fname))
            except RuntimeError as err:
                utils.logger.warning(f"Skip loading generator checkpoint due to mismatch: {err}")
        else:
            utils.logger.info('generator state file is not found')

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class Discriminator(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, hidden_dims, state_fname, device) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.device = device
        input_dim = maxLength * num_vocabs
        dims = [input_dim] + list(hidden_dims) + [1]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1], device=self.device))
            if idx < len(dims) - 2:
                layers.append(torch.nn.LeakyReLU(0.2))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        X = X.to(self.device)
        X = X.reshape(X.shape[0], -1)
        logits = self.net(X)
        return torch.sigmoid(logits)

    def loadState(self):
        if os.path.isfile(self.state_fname):
            try:
                self.load_state_dict(torch.load(self.state_fname))
            except RuntimeError as err:
                utils.logger.warning(f"Skip loading discriminator checkpoint due to mismatch: {err}")
        else:
            utils.logger.info('discriminator state file is not found')

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class GAN(object):
    def __init__(self, maxLength, num_vocabs, latent_dim, generator_hidden, discriminator_hidden, label_smoothing, generator_state_fname, discriminator_state_fname, device) -> None:
        self.device = device
        self.label_smoothing = label_smoothing
        self.generator = Generator(maxLength, num_vocabs, latent_dim, generator_hidden,
                                   generator_state_fname, device)
        self.discriminator = Discriminator(maxLength, num_vocabs, discriminator_hidden,
                                           discriminator_state_fname, device)
        self.criterion = torch.nn.BCELoss()

    def sample(self, nSample, temperature=1.0, greedy=False):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn((nSample, self.generator.latent_dim), device=self.device)
            probs = self.generator(noise, temperature)
            if greedy:
                token_ids = torch.argmax(probs, dim=-1)
            else:
                flat_probs = probs.view(-1, probs.shape[-1])
                samples = torch.distributions.Categorical(probs=flat_probs).sample()
                token_ids = samples.view(nSample, self.generator.maxLength)
            token_ids = token_ids + 2
        return token_ids.cpu(), probs.cpu()

    def latent_space_quality(self, nSample, tokenizer, is_valid_fn=None):
        if is_valid_fn is None:
            is_valid_fn = utils.isValidSmiles
        numVectors, _ = self.sample(nSample, temperature=1.0, greedy=False)
        smilesStrs = tokenizer.getSmiles(numVectors)
        validSmilesStrs = [sm for sm in smilesStrs if is_valid_fn(sm)]
        return len(validSmilesStrs), len(set(validSmilesStrs))

    def trainModel(self, dataloader, generatorOptimizer, discriminatorOptimizer, generatorScheduler,
                   discriminatorScheduler, nepoch, tokenizer, printInterval, is_valid_fn=None):
        self.generator.loadState()
        self.discriminator.loadState()
        best_valid = 0
        numSample = 100
        if is_valid_fn is None:
            is_valid_fn = utils.isValidSmiles
        for epoch in range(1, nepoch + 1):
            g_loss_epoch, d_loss_epoch = 0.0, 0.0
            g_loss_interval, d_loss_interval = [], []
            valid_list, unique_list = [], []
            for nBatch, real_X in enumerate(dataloader, 1):
                real_X = real_X.to(self.device)
                batch_size = real_X.shape[0]
                valid_labels = torch.full((batch_size, 1), 1.0 - self.label_smoothing, device=self.device)
                fake_labels = torch.zeros((batch_size, 1), device=self.device)

                noise = torch.randn((batch_size, self.generator.latent_dim), device=self.device)
                fake_X = self.generator(noise).detach()

                discriminatorOptimizer.zero_grad()
                real_pred = self.discriminator(real_X)
                fake_pred = self.discriminator(fake_X)
                d_loss_real = self.criterion(real_pred, valid_labels)
                d_loss_fake = self.criterion(fake_pred, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                discriminatorOptimizer.step()

                noise = torch.randn((batch_size, self.generator.latent_dim), device=self.device)
                generatorOptimizer.zero_grad()
                generated = self.generator(noise)
                generator_pred = self.discriminator(generated)
                generator_targets = torch.full((batch_size, 1), 1.0 - self.label_smoothing, device=self.device)
                g_loss = self.criterion(generator_pred, generator_targets)
                g_loss.backward()
                generatorOptimizer.step()

                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
                g_loss_interval.append(g_loss.item())
                d_loss_interval.append(d_loss.item())

                if nBatch == 1 or nBatch % printInterval == 0:
                    valid_cnt, unique_cnt = self.latent_space_quality(numSample, tokenizer, is_valid_fn=is_valid_fn)
                    valid_list.append(valid_cnt)
                    unique_list.append(unique_cnt)
                    avg_d_loss = sum(d_loss_interval) / len(d_loss_interval)
                    avg_g_loss = sum(g_loss_interval) / len(g_loss_interval)
                    utils.logger.info(
                        f"Epoch {epoch:4d} & Batch {nBatch:4d}: D_Loss= {avg_d_loss:.5e} G_Loss= {avg_g_loss:.5e} "
                        f"Valid= {valid_cnt:3d}/{numSample:3d} Unique= {unique_cnt:3d}"
                    )
                    g_loss_interval.clear()
                    d_loss_interval.clear()
                    if valid_cnt > best_valid:
                        self.generator.saveState()
                        self.discriminator.saveState()
                        best_valid = valid_cnt

            generatorScheduler.step()
            discriminatorScheduler.step()
            mean_valid = sum(valid_list) / len(valid_list) if valid_list else 0.0
            mean_unique = sum(unique_list) / len(unique_list) if unique_list else 0.0
            avg_d_loss_epoch = d_loss_epoch / nBatch
            avg_g_loss_epoch = g_loss_epoch / nBatch
            utils.logger.info(
                f"Epoch {epoch:4d}: D_Loss= {avg_d_loss_epoch:.5e} G_Loss= {avg_g_loss_epoch:.5e} "
                f"Valid= {mean_valid:.0f}/{numSample:3d} Unique= {mean_unique:.0f}"
            )


from dataset import SmilesDataset
import utils
import torch
import rnn
import lstm
import transformer
import mamba
import vae
import gan
import flow
import argparse
import diffusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('--data-type', type=str, choices=['smiles', 'protein'], default=utils.config['data_variant'], help='Select dataset/tokenizer: smiles or protein')
    parser.add_argument('-i', '--printInterval', type=int, default=100)
    parser.add_argument('--gpu', action='store_true', default=False)
    args = parser.parse_args()
    model_type = args.model
    printInterval = args.printInterval
    device_type = 'cpu'
    if args.gpu:
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
    device = torch.device(device_type)
    if args.data_type != utils.config['data_variant']:
        utils.switch_data_variant(args.data_type)
    tokenizer = utils.get_tokenizer()
    valid_fn = utils.isValidSmiles if utils.config['data_variant'] == 'smiles' else (lambda s: bool(s))
    utils.logger.info(f'Device={device_type}')
    utils.logger.info(f'Tokens: {list(tokenizer.tokensDict.keys())}')
    smilesDataset = SmilesDataset(
        utils.config['fname_dataset'], tokenizer, utils.config['maxLength'])
    if model_type == 'rnn':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.collate_fn)
        rnn_model = rnn.RNN(
            **utils.config['rnn_param'], state_fname=utils.config['fname_rnn_parameters'], device=device)
        for layer_index, (name, layer) in enumerate(rnn_model.named_parameters()):
            utils.logger.info(f'Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        optimizer = torch.optim.Adam(
            rnn_model.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.95)
        valid_fn = utils.isValidSmiles if utils.config['data_variant'] == 'smiles' else (lambda s: bool(s))
        rnn_model.trainModel(smilesDataloader, optimizer, scheduler,
                             utils.config['num_epoch'], utils.config['maxLength'], tokenizer, printInterval, is_valid_fn=valid_fn)
    elif model_type == 'lstm':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.collate_fn)
        lstm_model = lstm.LSTM(
            **utils.config['lstm_param'], state_fname=utils.config['fname_lstm_parameters'], device=device)
        for layer_index, (name, layer) in enumerate(lstm_model.named_parameters()):
            utils.logger.info(f'Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        optimizer = torch.optim.Adam(
            lstm_model.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.95)
        lstm_model.trainModel(smilesDataloader, optimizer, scheduler,
                             utils.config['num_epoch'], utils.config['maxLength'], tokenizer, printInterval, is_valid_fn=valid_fn)
    elif model_type == 'transformer':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.collate_fn)
        transformer_model = transformer.Transformer(
            **utils.config['transformer_param'], state_fname=utils.config['fname_transformer_parameters'], device=device)
        for layer_index, (name, layer) in enumerate(transformer_model.named_parameters()):
            utils.logger.info(f'Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        optimizer = torch.optim.Adam(
            transformer_model.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.95)
        transformer_model.trainModel(smilesDataloader, optimizer, scheduler,
                             utils.config['num_epoch'], utils.config['maxLength'], tokenizer, printInterval, is_valid_fn=valid_fn)
    elif model_type == 'mamba':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.collate_fn)
        mamba_model = mamba.Mamba(
            **utils.config['mamba_param'], state_fname=utils.config['fname_mamba_parameters'], device=device)
        for layer_index, (name, layer) in enumerate(mamba_model.named_parameters()):
            utils.logger.info(f'Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        optimizer = torch.optim.Adam(
            mamba_model.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.95)
        mamba_model.trainModel(smilesDataloader, optimizer, scheduler,
                             utils.config['num_epoch'], utils.config['maxLength'], tokenizer, printInterval, is_valid_fn=valid_fn)
    elif model_type == 'vae':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.one_hot_collate_fn)
        vae_model = vae.VAE(**utils.config['vae_param'], encoder_state_fname=utils.config['fname_vae_encoder_parameters'],
                            decoder_state_fname=utils.config['fname_vae_decoder_parameters'], device=device)
        for layer_index, (name, layer) in enumerate(vae_model.encoder.named_parameters()):
            utils.logger.info(f'Encoder Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        for layer_index, (name, layer) in enumerate(vae_model.decoder.named_parameters()):
            utils.logger.info(f'Decoder Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        encoderOptimizer = torch.optim.Adam(
            vae_model.encoder.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        decoderOptimizer = torch.optim.Adam(
            vae_model.decoder.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        encoderScheduler = torch.optim.lr_scheduler.StepLR(
            encoderOptimizer, step_size=1, gamma=0.95)
        decoderScheduler = torch.optim.lr_scheduler.StepLR(
            decoderOptimizer, step_size=1, gamma=0.95)
        vae_model.trainModel(smilesDataloader, encoderOptimizer, decoderOptimizer, encoderScheduler,
                             decoderScheduler, 1.0, utils.config['num_epoch'], tokenizer, printInterval, is_valid_fn=valid_fn)
    elif model_type == 'gan':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.one_hot_collate_fn)
        gan_model = gan.GAN(**utils.config['gan_param'],
                            generator_state_fname=utils.config['fname_gan_generator_parameters'],
                            discriminator_state_fname=utils.config['fname_gan_discriminator_parameters'],
                            device=device)
        for layer_index, (name, layer) in enumerate(gan_model.generator.named_parameters()):
            utils.logger.info(f'Generator Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        for layer_index, (name, layer) in enumerate(gan_model.discriminator.named_parameters()):
            utils.logger.info(f'Discriminator Layer {layer_index+1:02d}: name={name}, shape={list(layer.shape)}, dtype={layer.dtype}, grad={layer.requires_grad}, device={layer.device}')
        generatorOptimizer = torch.optim.Adam(
            gan_model.generator.parameters(), lr=utils.config['lr'], betas=(0.5, 0.999))
        discriminatorOptimizer = torch.optim.Adam(
            gan_model.discriminator.parameters(), lr=utils.config['lr'], betas=(0.5, 0.999))
        generatorScheduler = torch.optim.lr_scheduler.StepLR(
            generatorOptimizer, step_size=1, gamma=0.95)
        discriminatorScheduler = torch.optim.lr_scheduler.StepLR(
            discriminatorOptimizer, step_size=1, gamma=0.95)
        gan_model.trainModel(smilesDataloader, generatorOptimizer, discriminatorOptimizer, generatorScheduler,
                             discriminatorScheduler, utils.config['num_epoch'], tokenizer, printInterval, is_valid_fn=valid_fn)
    elif model_type == 'diffusion':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.one_hot_collate_fn)
        diffusion_model = diffusion.Diffusion(**utils.config['diffusion_param'], state_fname=utils.config['fname_diffusion_parameters'], device=device)
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        diffusion_model.trainModel(smilesDataloader, optimizer, scheduler, utils.config['num_epoch'], tokenizer, printInterval, is_valid_fn=valid_fn)
    elif model_type == 'flow':
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset, batch_size=utils.config['batch_size'], shuffle=True, num_workers=4, collate_fn=smilesDataset.one_hot_collate_fn)
        flow_model = flow.Flow(**utils.config['flow_param'], state_fname=utils.config['fname_flow_parameters'], device=device)
        optimizer = torch.optim.Adam(flow_model.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        flow_model.trainModel(smilesDataloader, optimizer, scheduler, utils.config['num_epoch'], tokenizer, printInterval, is_valid_fn=valid_fn)

if __name__ == '__main__':
    main()
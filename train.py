import click
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import logging
import coloredlogs


from torch.optim import Adam
from Optim import ScheduledOptim
# from optim import optim4GPU
import torch
import torch.nn as nn
from dataset import DataLoader
from dataset import ParallelLanguageDataset
from dataset import WordVocab
from model import ScoreTransformer
from dataset import all_tokens
from dataset import collate_mlm

@click.command()
@click.argument('num_epochs', type=int,  default=10)
@click.argument('max_seq_length', type=int,  default=2500)
@click.argument('d_model', type=int,  default=256)
@click.argument('num_encoder_layers', type=int,  default=4)
@click.argument('num_decoder_layers', type=int,  default=4)
@click.argument('dim_feedforward', type=int,  default=2048)
@click.argument('nhead', type=int,  default=4)
@click.argument('pos_dropout', type=float,  default=0.1)
@click.argument('trans_dropout', type=float,  default=0.1)
@click.argument('n_warmup_steps', type=int,  default=4000)
@click.option('--n_bar','-b', type=int,  default=16)
@click.option('--output_folder','-o', type=str,  default='./')
@click.option('--device','-d', type=str,  default='cuda')
def main(**kwargs):
    logger = logging.getLogger(__name__)
    output_folder = kwargs['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    bar_num = kwargs['n_bar']
    device = torch.device(kwargs['device'])
    print(device)
    print(f'bar num is {bar_num}')
    print(f'output folder is {output_folder}')

    logger.handlers = []
    logfile = output_folder + '/logging.log'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S', filename=logfile,filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)

    coloredlogs.install(level='INFO', logger=logger, isatty=True)

    
   

    vocab = WordVocab(all_tokens)
    event_folder = '../dataset/lmd/lmd_separate_event/'
    window_size = int(bar_num / 2)
    max_token_length = 2400
    train_ratio = 0.5
    valid_ratio = 0.05

    total_mask_ratio, structure_mask_ratio, \
    duration_mask_ratio, pitch_mask_ratio, \
    control_mask_ratio, header_mask_ratio = [.15,.3,.0,.0,.0,.0]

    train_dataset = ParallelLanguageDataset(event_folder, vocab, 0,train_ratio, max_token_length, window_size, total_mask_ratio, structure_mask_ratio, \
    duration_mask_ratio, pitch_mask_ratio, \
    control_mask_ratio, header_mask_ratio)

    valid_dataset = ParallelLanguageDataset(event_folder, vocab, train_ratio, train_ratio+valid_ratio, max_token_length, window_size, total_mask_ratio, structure_mask_ratio, \
    duration_mask_ratio, pitch_mask_ratio, \
    control_mask_ratio, header_mask_ratio)

    train_data_loader = DataLoader(train_dataset, batch_size=1, collate_fn=lambda batch: collate_mlm(batch))
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=lambda batch: collate_mlm(batch))

    # Set batch_size=1 because all the batching is handled in the ParallelLanguageDataset class

    model = ScoreTransformer(vocab.vocab_size, kwargs['d_model'], kwargs['nhead'], kwargs['num_encoder_layers'],
                                kwargs['num_decoder_layers'], kwargs['dim_feedforward'], kwargs['max_seq_length'],
                                kwargs['pos_dropout'], kwargs['trans_dropout'])


    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    model.to(device)


    # Use Xavier normal initialization in the transformer
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    optim = Adam(model.parameters(),lr=0.0001)
    # optim = ScheduledOptim(
    #     Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    #     kwargs['d_model'], kwargs['n_warmup_steps'])
    # total_steps = len(train_data_loader) * kwargs['num_epochs']
    # optim = optim4GPU(model, total_steps)

    # Use cross entropy loss, ignoring any padding]
    weight = torch.ones(vocab.vocab_size,device=device)
    weight[1] = 0.2
    # logger.info(weight.get_device())
    criterion = nn.CrossEntropyLoss(ignore_index=0,weight=weight)
    train_losses,valid_losses = train(train_data_loader, valid_data_loader, model, device,optim, criterion, logger,kwargs['num_epochs'], vocab)
    logger.info(f'training losses is {train_losses}'
                f'validation losses is {valid_losses}')


def accuracy(outputs,targets,vocab):
    # define total accuracy, structure accuracy, control accuracy,
    # duration accuracy, pitch accuracy
    with torch.no_grad():
        print('\n')
        total_accuracy = 0
        structure_accuracy = 0
        structure_length = 0
        control_accuracy = 0
        control_length = 0
        duration_accuracy = 0
        duration_length = 0
        pitch_accuracy = 0
        pitch_length = 0
        program_accuracy = 0
        program_length = 0
        eos_accuracy = 0
        eos_length = 0
        total_length = 0
        # total_length = int(outputs.size()[0] * outputs.size()[1])
        generated_output = []
        target_output = []

        for i, output in enumerate(outputs):

            for position, token_idx in enumerate(torch.argmax(output, axis=1)):
                # output_classes = vocab.get_token_classes(token_idx)

                output_token = vocab.index2char(token_idx.item())


                target_idx = targets[i][position].item()
                target_token = vocab.index2char(target_idx)

                if i == 0:
                    generated_output.append(output_token)
                    target_output.append(target_token)

                if target_idx == vocab.pad_index:
                    continue

                target_classes = vocab.get_token_classes(target_idx)


                if target_classes == 'structure':
                    structure_accuracy += token_idx.item() == target_idx
                    structure_length += 1
                if target_classes == 'pitch':
                    pitch_accuracy += token_idx.item() == target_idx
                    pitch_length += 1
                if target_classes == 'duration':
                    duration_accuracy += token_idx.item() == target_idx
                    duration_length += 1
                if target_classes == 'control':
                    control_accuracy += token_idx.item() == target_idx
                    control_length += 1
                if target_classes == 'program':
                    program_accuracy += token_idx.item() == target_idx
                    program_length += 1
                if target_classes == 'eos':
                    eos_accuracy += token_idx.item() == target_idx
                    eos_length += 1

                total_accuracy += token_idx.item() == target_idx
                total_length += 1


        total_accuracy /= total_length

        program_accuracy = program_accuracy / program_length if program_length > 0 else 0
        control_accuracy = control_accuracy / control_length if control_length > 0 else 0
        structure_accuracy = structure_accuracy / structure_length if structure_length > 0 else 0
        pitch_accuracy = pitch_accuracy / pitch_length if pitch_length > 0 else 0
        duration_accuracy = duration_accuracy / duration_length if duration_length > 0 else 0
        eos_accuracy = eos_accuracy / eos_length if eos_length > 0 else 0

        return {'total': total_accuracy,
                'program': program_accuracy,
                'control': control_accuracy,
                'structure': structure_accuracy,
                'pitch': pitch_accuracy,
                'duration': duration_accuracy,
                'eos': eos_accuracy}, generated_output, target_output



def train(train_loader, valid_loader, model,device, optim, criterion, logger,num_epochs, vocab):
    print_every = 100
    model.train()


    lowest_val = 1e9
    train_losses = []
    train_accuracies = {'total':[],
                        'pitch':[],
                        'duration':[],
                        'structure':[],
                        'control':[],
                        'program':[],
                        'eos':[]}
    val_losses = []
    total_step = 0
    for epoch in range(num_epochs):
        # pbar = tqdm(total=print_every, leave=False)
        total_loss = 0
        every_print_accuracy = {'total':0,
                        'pitch':0,
                        'duration':0,
                        'structure':0,
                        'control':0,
                        'program':0,
                        'eos':0}



        # Shuffle batches every epoch
        # train_loader.dataset.shuffle_batches()
        for step, data in enumerate(iter(train_loader)):
            total_step += 1

            # Send the batches and key_padding_masks to gpu
            src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
            tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
            tgt_out = data['target_out'].to(device)
            memory_key_padding_mask = src_key_padding_mask.clone()

            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)

            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)

            # Forward
            optim.zero_grad()
            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
            # logger.info(src.size())
            # logger.info(tgt.size())
            # logger.info("outside model, output_size:", outputs.size())

            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            # Backpropagate and update optim
            loss.backward()

            # optim.step_and_update_lr()
            optim.step()

            total_loss += loss.item()
            train_losses.append((step, loss.item()))
            # accuracies = accuracy(outputs,tgt_out,vocab)
            # total_accuracy += accuracies['total']

            # for key in train_accuracies.keys():
            #     train_accuracies[key].append((step,accuracies[key]))

            # logger.info(f'loss is {loss}')
            # logger.info(f'total accuracy is {accuracies["total"]}')

            # pbar.update(1)
            if step % print_every == print_every - 1:
                # pbar.close()
                times = int((step / (print_every - 1)))

                accuracies,generated_output, target_output = accuracy(outputs,tgt_out,vocab)
                every_print_accuracy['total'] += accuracies['total']
                every_print_accuracy['pitch'] += accuracies['pitch']
                every_print_accuracy['duration'] += accuracies['duration']
                every_print_accuracy['structure'] += accuracies['structure']
                every_print_accuracy['control'] += accuracies['control']
                every_print_accuracy['program'] += accuracies['program']
                every_print_accuracy['eos'] += accuracies['eos']

                # lr = optim.get_lr()

                # logger.info(f'loss is {loss}')
                # logger.info(f'total accuracy is {accuracies["total"]}')
                logger.info(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \n '
                      f'Train Loss: {total_loss / print_every} \t Accuracy {accuracies["total"]} \n'
                             f'structure accuracy : {every_print_accuracy["structure"] / times} \n'
                            f'duration accuracy : {every_print_accuracy["duration"] / times} \n'
                            f'pitch accuracy : {every_print_accuracy["pitch"] / times} \n'
                            f'program accuracy : {every_print_accuracy["program"] / times} \n'
                            f'generated output: {generated_output[:50]} \n'
                            f'target output: {target_output[:50]} \n'
                            f'target size is {len(target_output)} \n'
                            f'src size is {src.size()}')


                total_loss = 0

                # pbar = tqdm(total=print_every, leave=False)
        total_times = int(step / (print_every-1))
        for key in train_accuracies.keys():
            train_accuracies[key].append(every_print_accuracy[key] / total_times)
        logger.info(f'Epoch [{epoch + 1} / {num_epochs} end] \t ')
        for key in train_accuracies.keys():
            logger.info(f' {key} accuracy is {train_accuracies[key][epoch]} \t ')


        # Validate every epoch
        # pbar.close()
        val_loss = validate(valid_loader, model, criterion, device)
        val_losses.append((total_step, val_loss))
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model, 'output/transformer.pth')
        logger.info(f'Val Loss: {val_loss}')
    return train_losses , val_losses


def validate(valid_loader, model, criterion, device):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    total_loss = 0

    for data in iter(valid_loader):

        # Send the batches and key_padding_masks to gpu
        src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
        tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
        tgt_out = data['target_out'].to(device).contiguous()
        memory_key_padding_mask = src_key_padding_mask.clone()

        tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)
        with torch.no_grad():

            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            total_loss += loss.item()
            # pbar.update(1)

    # pbar.close()
    model.train()
    return total_loss / len(valid_loader)


def gen_nopeek_mask(length):
    """
     Returns the nopeek mask
             Parameters:
                     length (int): Number of tokens in each sentence in the target batch
             Returns:
                     mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                      [0., 0., -inf],
                                                      [0., 0., 0.]]
     """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


if __name__ == "__main__":
    main()

from pathlib import Path
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import logging
import coloredlogs
import pickle

from torch.optim import Adam
from Optim import ScheduledOptim
# from optim import optim4GPU
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ParallelLanguageDataset
from dataset import WordVocab
from model import ScoreTransformer
from dataset import all_tokens
from dataset import collate_mlm
from torch.nn import functional as F

# event_folder = '/home/ruiguo/dataset/lmd/lmd_separate_event'
# train_ratio = 0.05
# valid_ratio = 0.01
# test_ratio = 0.01
# num_epochs = 2
# device='cuda:1'
# max_token_length = 2400
span_ratio_separately_each_epoch = np.array([[1, 0, 0], [.5, .5, 0],
                                             [.25, .75, 0], [.25, .5, .25],
                                             [.25, .25, .5]])


def get_args(default='.'):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--platform', default='local', type=str,
                        help="platform to run the code")
    # parser.add_argument('-v', '--valid_ratio', default=0.1, type=float,
    #                     help="valid data ratio")
    # parser.add_argument('-s', '--test_ratio', default=0.1, type=float,
    #                     help="test data ratio")
    parser.add_argument('-e', '--num_epochs', default=10, type=int,
                        help="number of epoch")
    parser.add_argument('-d', '--is_debug', default=False, type=bool,
                        help="debug or not")
    parser.add_argument('-c', '--checkpoint_dir', default="", type=str,
                        help="checkpoint dir")
    #
    # parser.add_argument('-d', '--device', default='cuda:1', type=str,
    #                     help="gpu name")
    # parser.add_argument('-m', '--max_token_length', default=2200, type=int,
    #                     help="max token length")
    # parser.add_argument('-f', '--event_folder', default='/home/ruiguo/dataset/lmd/lmd_separate_event', type=str,
    #                     help="max token length")
    #
    # parser.add_argument('-n', '--tension_folder', default='/home/ruiguo/dataset/lmd/lmd_tension_three_tracks', type=str,
    #                     help="tension folder")
    # parser.add_argument('-j', '--train_jointly', default='True', type=str,
    #                     help="train jointly or not")
    # parser.add_argument('-w', '--eos_weight', default=0.3, type=float,
    #                     help="eos weight")

    return parser.parse_args()


span_ratio_separately_each_epoch = np.array([[1, 0, 0], [.5, .5, 0],
                                             [.25, .75, 0], [.25, .5, .25],
                                             [.25, .25, .5]])


def phi(rt, ri):
    rt = rt.type(torch.float32)
    return torch.square(rt - ri)


def soft_label(total_label_num, target_index_range):
    target_index_length = target_index_range[1] - target_index_range[0] + 1
    output_weights = torch.zeros(total_label_num, total_label_num)
    weights = nn.functional.softmax(-phi(torch.unsqueeze(torch.arange(target_index_length), dim=1),
                                         torch.arange(target_index_length)), 0)
    output_weights[target_index_range[0]:target_index_range[1] + 1,
    target_index_range[0]:target_index_range[1] + 1] = weights
    return output_weights


class OrdinalLoss(nn.Module):
    def __init__(self, target_index_range, vocab_size, device, **kwargs):
        super().__init__()
        # a tuple to delineate the target index
        self.weights = soft_label(vocab_size, target_index_range).to(device)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x, target):
        logsoftmax = self.logsoftmax(x)
        target_array = -self.weights[target]
        return torch.mean(torch.sum(torch.multiply(logsoftmax, target_array), axis=1))


def main(**kwargs):
    # device = torch.device(kwargs['device'])
    # print(f'device is {device}')
    args = get_args()

    global platform, num_epochs,is_debug

    # event_folder = args.event_folder
    # tension_folder = args.tension_folder
    # train_ratio = args.train_ratio
    # valid_ratio = args.valid_ratio
    # test_ratio = args.test_ratio
    num_epochs = args.num_epochs
    platform = args.platform
    is_debug = args.is_debug
    checkpoint_dir = args.checkpoint_dir

    # max_token_length = 2200
    # train_jointly = args.train_jointly
    # eos_weight = 0.3

    train_jointly = True
    # if train_jointly == 'True':
    #     train_jointly = True
    # else:
    #     train_jointly = False

    # print(f'event_folder is {event_folder}')
    # print(f'tension_folder is {tension_folder}')
    # print(f'train_ratio is {train_ratio}')
    # print(f'test_ratio is {test_ratio}')
    # print(f'valid_ratio is {valid_ratio}')
    #
    # print(f'num_epochs is {num_epochs}')
    print(f'num_epochs is {num_epochs}')
    print(f'platform is {platform}')
    if checkpoint_dir:
        print(f'checkpoint dir is {checkpoint_dir}')

    # print(f'max_token_length is {max_token_length}')
    # print(f'train jointly is {train_jointly}')

    config = {"batch_size": tune.grid_search([2]),

              "span_lengths": tune.grid_search([3]),
              "span_ratio_jointly": tune.grid_search([0.5]),
              "eos_weight": tune.grid_search([0.8]),
              # "train_jointly": tune.grid_search([True]),
              'd_model': tune.grid_search([512]),
              'num_encoder_layers': tune.grid_search([4]),
              'loss_num': tune.grid_search([2]),
              # 'num_decoder_layers': tune.grid_search([4]),
              # 'dim_feedforward': tune.grid_search([2048]),
              'nhead': tune.grid_search([8]),
              # 'max_seq_length': tune.grid_search([2400]),
              # 'pos_dropout': tune.grid_search([0.1]),
              # 'trans_dropout': tune.grid_search([0.1]),
              # 'total_mask_ratio': tune.grid_search([.15]),
              # 'structure_mask_ratio': tune.grid_search([.3]),
              # 'duration_mask_ratio': tune.grid_search([.3]),
              # 'pitch_mask_ratio': tune.grid_search([.3]),
              # 'control_mask_ratio': tune.grid_search([0]),
              # 'header_mask_ratio': tune.grid_search([.3]),
              # 'bar_num': tune.grid_search([16])
              }

    gpus_per_trial = 2
    cpus_per_trial = 8
    num_samples = 1

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=num_epochs,
        grace_period=8,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=[
            "train_loss",
            "train_accuracy",
            "lr",
            "loss",
            "val_loss",
            "val_accuracy",
            "pitch_accuracy",
            "duration_accuracy",
            "structure_accuracy",
            "tempo_accuracy",
            "time_signature_accuracy",
            "program_accuracy",
            "eos_accuracy",
            "track_control_accuracy",
            "bar_control_accuracy",
            "density_accuracy",
            "occupation_accuracy",
            "polyphony_accuracy",
            "pitch_register_accuracy",
            "tensile_accuracy",
            "diameter_accuracy",
            "key_accuracy"])
    if is_debug:
        train_ray_tune(config,checkpoint_dir=checkpoint_dir)
    else:
        result = tune.run(
            train_ray_tune,
            resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir='./ray_result/'
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["total"]))

        # vocab = WordVocab(all_tokens)
        #
        # best_trained_model = ScoreTransformer(vocab.vocab_size,
        #                                       best_trial.config['d_model'],
        #                                       best_trial.config['nhead'],
        #                                       best_trial.config['num_encoder_layers'],
        #                                       best_trial.config['num_encoder_layers'],
        #                                       2048, 2400,
        #                                       0.1, 0.1)
        #
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     best_trained_model = nn.DataParallel(best_trained_model)
        #
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #
        # best_trained_model.to(device)
        # best_checkpoint_dir = best_trial.checkpoint.value
        # window_size = int(16 / 2)
        #
        #
        #
        # if platform == 'local':
        #     folder_prefix = '/home/ruiguo/'
        # else:
        #     folder_prefix = '/content/drive/MyDrive/'
        # test_batches = pickle.load(open(folder_prefix + 'score_transformer/test_batches_0_0_5_new_bins', 'rb'))
        # test_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/test_batch_lengths_0_0_5_new_bins', 'rb'))
        #
        # test_dataset = ParallelLanguageDataset('', '',
        #                                        vocab, 0,
        #                                        0,
        #                                        2400,
        #                                        16,
        #                                        test_batches,
        #                                        test_batch_lengths,
        #                                        .15,
        #                                        .3,
        #                                        .3,
        #                                        .3,
        #                                        0,
        #                                        .3,
        #                                        3,
        #                                        0.5,
        #                                        span_ratio_separately_each_epoch,
        #                                        True)
        #
        # test_data_loader = DataLoader(test_dataset, batch_size=best_trial.config['batch_size'],
        #                               collate_fn=lambda batch: collate_mlm(batch))
        #
        # logger = logging_config(best_checkpoint_dir, append=True)
        #
        # logger.info(f'test data size is {len(test_data_loader)}')
        #
        # model_state, optimizer_state = torch.load(os.path.join(
        #     best_checkpoint_dir, "checkpoint"))
        # best_trained_model.load_state_dict(model_state)
        #
        #
        # weight = torch.ones(vocab.vocab_size, device=device)
        # weight[1] = best_trial.config['eos_weight']
        #
        # weight[7:18] = best_trial.config['control_weight']
        # weight[146:234] = best_trial.config['control_weight']
        #
        # criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight)
        #
        # test_loss, test_acc = test_loss_accuracy(test_data_loader,
        #                                          best_trained_model,
        #                                          criterion,
        #                                          device,
        #                                          vocab,
        #                                          logger)
        # logger.info(f'Best trial test set loss: {test_loss}')
        # logger.info(f'Best trial test set accuracy: {test_acc["total_accuracy"]}')
        #

def logging_config(output_folder, append=False):
    logger = logging.getLogger(__name__)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    logger.handlers = []
    logfile = output_folder + '/logging.log'
    print(f'log file is {logfile}')

    if append is True:
        filemode = 'a'
    else:
        filemode = 'w'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S', filename=logfile, filemode=filemode)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)

    coloredlogs.install(level='INFO', logger=logger, isatty=True)
    logger.info('create a logger file')
    return logger


def train_ray_tune(config, checkpoint_dir=None):
    if not is_debug:
        print(f'config is {config}')
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    current_folder = os.getcwd()
    logger = logging_config(current_folder)
    vocab = WordVocab(all_tokens)

    if is_debug:
        for key in config.keys():
            config[key] = config[key]['grid_search'][0]

    for key in config.keys():
        logger.info(f'{key} is {config[key]}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f'bar num is {16}')
    logger.info(f'output folder is {current_folder}')
    logger.info(f'vocab size is {vocab.vocab_size}')
    logger.info(f'platform is {platform}')

    model = ScoreTransformer(vocab.vocab_size, config['d_model'], config['nhead'], config['num_encoder_layers'],
                             config['num_encoder_layers'], 2048, 2400,
                             0.1, 0.1)

    weight = torch.ones(vocab.vocab_size, device=device)

    weight[1] = config['eos_weight']

    if config['loss_num'] == 1:
        weight[11:18] = 0
        weight[170:234] = 0

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    optim = Adam(model.parameters(), lr=0.0001)

    if is_debug and checkpoint_dir:

        model_state, optimizer_state = torch.load(os.path.join(
            checkpoint_dir,
            "checkpoint"))

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)



    window_size = int(16 / 2)

    if platform == 'local':
        folder_prefix = '/home/ruiguo/'
    else:
        folder_prefix = '/content/drive/MyDrive/'

    train_batches = pickle.load(open(folder_prefix + 'score_transformer/train_batches_0_5_new_bins', 'rb'))
    train_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/train_batch_lengths_0_5_new_bins', 'rb'))

    valid_batches = pickle.load(open(folder_prefix + 'score_transformer/valid_batches_0_0_5_new_bins', 'rb'))
    valid_batch_lengths = pickle.load(
        open(folder_prefix + 'score_transformer/valid_batch_lengths_0_0_5_new_bins', 'rb'))
    logger.info(f'train batches file is  {folder_prefix + "score_transformer/train_batches_0_5_new_bins"}')
    logger.info(f'valid batches file is  {folder_prefix + "score_transformer/valid_batches_0_0_5_new_bins"}')

    logger.info(f'train batch length is {len(train_batches)}')
    logger.info(f'valid batch length is {len(valid_batches)}')
    # print(f'test batch length is {len(test_batches)}')

    train_dataset = ParallelLanguageDataset('',
                                            '',
                                            vocab,
                                            0, 0,
                                            2400,
                                            window_size,
                                            train_batches,
                                            train_batch_lengths,
                                            config['batch_size'],
                                            .15,
                                            .3,
                                            .3,
                                            .3,
                                            .9,
                                            .3,
                                            3,
                                            0.5,
                                            span_ratio_separately_each_epoch,
                                            True)

    valid_dataset = ParallelLanguageDataset('',
                                            '',
                                            vocab, 0,
                                            0,
                                            2400,
                                            window_size,
                                            valid_batches,
                                            valid_batch_lengths,
                                            config['batch_size'],
                                            .15,
                                            .3,
                                            .3,
                                            .3,
                                            .9,
                                            .3,
                                            3,
                                            0.5,
                                            span_ratio_separately_each_epoch,
                                            True)

    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                   collate_fn=lambda batch: collate_mlm(batch),pin_memory=True,num_workers=1)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                                   collate_fn=lambda batch: collate_mlm(batch),pin_memory=True,num_workers=1)

    #
    # logger.info(f'train data size is {len(train_data_loader)}')

    # Set batch_size=1 because all the batching is handled in the ParallelLanguageDataset class

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    # Use Xavier normal initialization in the transformer

    # optim = ScheduledOptim(
    #     Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
    #     kwargs['d_model'], kwargs['n_warmup_steps'])
    # total_steps = len(train_data_loader) * kwargs['num_epochs']
    # optim = optim4GPU(model, total_steps)

    # Use cross entropy loss, ignoring any padding]

    # logger.info(weight.get_device())

    if is_debug and checkpoint_dir:

        optim.load_state_dict(optimizer_state)

        # for state in optim.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.to(device)

        weight[1] = 1

    ce_loss = nn.CrossEntropyLoss(ignore_index=0, weight=weight)

    tempo_loss = OrdinalLoss((11, 17), vocab.vocab_size,device=device)
    density_loss = OrdinalLoss((170, 179), vocab.vocab_size,device=device)
    occupation_loss = OrdinalLoss((180, 189), vocab.vocab_size,device=device)
    polyphony_loss = OrdinalLoss((190, 199), vocab.vocab_size,device=device)
    pitch_register_loss = OrdinalLoss((200, 209), vocab.vocab_size,device=device)
    tensile_loss = OrdinalLoss((210, 221), vocab.vocab_size,device=device)
    diameter_loss = OrdinalLoss((222, 233), vocab.vocab_size,device=device)

    print_every = 1000
    model.train()

    lowest_val = 1e9
    train_losses = []

    train_accuracies = {'total': 0,
                        'track_control': 0,
                        'bar_control': 0}

    for token_type in set(vocab.token_class_ranges.values()):
        train_accuracies[token_type] = 0

    # train_accuracies = {'total': [],
    #                     'pitch': [],
    #                     'duration': [],
    #                     'structure': [],
    #                     'time_signature': [],
    #                     'tempo': [],
    #                     'control': [],
    #                     'program': [],
    #                     'eos': []}
    val_losses = []
    total_step = 0
    lr = 0
    for epoch in range(8,num_epochs):
        if epoch == 3:
            weight = torch.ones(vocab.vocab_size, device=device)

            # weight[1] = config['eos_weight']
            if config['loss_num'] == 1:
                weight[11:18] = 0
                weight[170:234] = 0
            ce_loss = nn.CrossEntropyLoss(ignore_index=0, weight=weight)

        scheduler_optim = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=10, factor=0.5, min_lr=0.0000001, verbose=True)
        # pbar = tqdm(total=print_every, leave=False)
        total_loss = 0
        every_print_accuracy = {'total': 0,
                                'track_control': 0,
                                'bar_control': 0}
        for token_type in set(vocab.token_class_ranges.values()):
            every_print_accuracy[token_type] = 0
        # every_print_accuracy = {'total': 0,
        #                         'pitch': 0,
        #                         'duration': 0,
        #                         'structure': 0,
        #                         'tempo': 0,
        #                         'time_signature': 0,
        #                         'control': 0,
        #                         'program': 0,
        #                         'eos': 0}

        # Shuffle batches every epoch
        # train_loader.dataset.shuffle_batches()
        for step, data in enumerate(iter(train_data_loader)):
            total_step += 1

            # Send the batches and key_padding_masks to gpu
            src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
            tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
            tgt_out = data['target_out'].to(device)
            memory_key_padding_mask = src_key_padding_mask.clone()

            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)

            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1])
            tgt_mask = torch.tensor(
                np.repeat(np.expand_dims(tgt_mask, 0), memory_key_padding_mask.shape[0], axis=0)).float()

            tgt_mask = tgt_mask.to(device)

            # Forward
            optim.zero_grad()
            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
            # logger.info(src.size())
            # logger.info(tgt.size())
            # logger.info("outside model, output_size:", outputs.size())

            loss_input_1 = rearrange(outputs, 'b t v -> (b t) v')
            loss_input_2 = rearrange(tgt_out, 'b o -> (b o)')
            loss1 = ce_loss(loss_input_1, loss_input_2)

            if config['loss_num'] == 1:
                loss2 = tempo_loss(loss_input_1, loss_input_2)
                loss3 = density_loss(loss_input_1, loss_input_2)
                loss4 = occupation_loss(loss_input_1, loss_input_2)
                loss5 = polyphony_loss(loss_input_1, loss_input_2)
                loss6 =  pitch_register_loss(loss_input_1, loss_input_2)
                loss7 =  tensile_loss(loss_input_1, loss_input_2)
                loss8 = diameter_loss(loss_input_1, loss_input_2)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + \
                       loss7 + loss8
            else:
                loss = loss1

            # loss = ce_loss(loss_input_1, loss_input_2) + \
            #     tempo_loss(loss_input_1, loss_input_2) + \
            #     density_loss(loss_input_1, loss_input_2) + \
            #     occupation_loss(loss_input_1, loss_input_2) + \
            #     polyphony_loss(loss_input_1, loss_input_2) + \
            #     pitch_register_loss(loss_input_1, loss_input_2) + \
            #     tensile_loss(loss_input_1, loss_input_2) + \
            #     diameter_loss(loss_input_1, loss_input_2)



            # loss = ce_loss(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)')) +

            # Backpropagate and update optim
            loss.backward()

            # optim.step_and_update_lr()
            optim.step()

            total_loss += loss.item()
            train_losses.append(loss.item())
            # accuracies = accuracy(outputs,tgt_out,vocab)
            # total_accuracy += accuracies['total']

            # for key in train_accuracies.keys():
            #     train_accuracies[key].append((step,accuracies[key]))

            # logger.info(f'loss is {loss}')
            # logger.info(f'total accuracy is {accuracies["total"]}')

            # pbar.update(1)
            if step % print_every == print_every - 1:
                scheduler_optim.step(total_loss / print_every)
                # pbar.close()
                times = int((step / (print_every - 1)))

                src_token = []

                for i, output in enumerate(src[0]):
                    output_token = vocab.index2char(output.item())
                    src_token.append(output_token)


                accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)

                for token_type in accuracies.keys():
                    every_print_accuracy[token_type] += accuracies[token_type]

                # logger.info(f'loss is {loss}')
                # logger.info(f'total accuracy is {accuracies["total"]}')
                logger.info(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_data_loader)}] \n \
                            Train Loss: {total_loss / print_every} \n Total accuracy: {every_print_accuracy["total"] / times} \n')

                if lr != optim.param_groups[0]['lr']:
                    lr = optim.param_groups[0]['lr']
                    logger.info(f'learning rate is {lr}')

                # if config['loss_num'] == 1:
                #     print(f'ce loss is {loss1}')
                #     print(f'tempo loss is {loss2}')
                #     print(f'density loss is {loss3}')
                #     print(f'occupation loss is {loss4}')
                #     print(f'polyphony loss is {loss5}')
                #     print(f'pitch_register loss is {loss6}')
                #     print(f'tensile loss is {loss7}')
                #     print(f'diameter loss is {loss8}')
                # structure accuracy : {every_print_accuracy["structure"] / times} \n \
                # duration accuracy : {every_print_accuracy["duration"] / times} \n \
                # pitch accuracy : {every_print_accuracy["pitch"] / times} \n \
                # track control accuracy : {every_print_accuracy["track_control"] / times} \n \
                # bar control accuracy : {every_print_accuracy["bar_control"] / times} \n')
                # f'program accuracy : {every_print_accuracy["program"] / times} \n'
                # f'eos accuracy : {every_print_accuracy["eos"] / times} \n',
                # # f'tempo accuracy : {every_print_accuracy["tempo"] / times} \n'
                # # f'time signature accuracy : {every_print_accuracy["time_signature"] / times} \n'

                #
                # )
                for token_type in every_print_accuracy.keys():
                    logger.debug(f'{token_type} accuracy is {every_print_accuracy[token_type] / times}')
                # logger.info(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_data_loader)}] \n '
                #             f'Train Loss: {total_loss / print_every} \t Accuracy {accuracies["total"] / times} \n'

                logger.debug(f'input size is {src.size()} \n'
                             f'input is : {src_token[:50]} \n'
                             f'output size is {len(target_output)} \n'
                             f'generated output: {generated_output[:50]} \n'
                             f'target output: {target_output[:50]} \n'
                             )

                total_loss = 0

                # pbar = tqdm(total=print_every, leave=False)

        for token_type in train_accuracies.keys():
            train_accuracies[token_type] = (every_print_accuracy[token_type] / times)

        average_train_loss = np.mean(train_losses)
        logger.info(f'Epoch [{epoch + 1} / {num_epochs} end] \t '
                    f'average train losses is {np.mean(average_train_loss)} \t'
                    f'average train accuracy is {train_accuracies["total"]}')

        # Validate every epoch
        # pbar.close()
        logger.info(f'valid data size is {len(valid_data_loader)}')
        val_loss, val_accuracy = validate(valid_data_loader, model, config,ce_loss,tempo_loss,density_loss,occupation_loss,polyphony_loss,pitch_register_loss,tensile_loss,diameter_loss, device, vocab, logger)
        val_losses.append(val_loss)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            print(f'checkpoint_dir is {checkpoint_dir}')

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'epoch': epoch,
                        'loss': val_loss}, path)

        tune.report(train_loss=average_train_loss,
                    train_accuracy=train_accuracies['total'],
                    lr=lr,
                    loss=val_loss,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy['total'],
                    pitch_accuracy=val_accuracy['pitch'],
                    duration_accuracy=val_accuracy['duration'],
                    structure_accuracy=val_accuracy['structure'],
                    tempo_accuracy=val_accuracy['tempo'],
                    time_signature_accuracy=val_accuracy['time_signature'],
                    program_accuracy=val_accuracy['program'],
                    eos_accuracy=val_accuracy['eos'],
                    track_control_accuracy=val_accuracy['track_control'],
                    bar_control_accuracy=val_accuracy['bar_control'],
                    density_accuracy=val_accuracy['density'],
                    occupation_accuracy=val_accuracy['occupation'],
                    polyphony_accuracy=val_accuracy['polyphony'],
                    pitch_register_accuracy=val_accuracy['pitch_register'],
                    tensile_accuracy=val_accuracy['tensile'],
                    diameter_accuracy=val_accuracy['diameter'],
                    key_accuracy=val_accuracy['key'],
                    )

    logger.info("Finished Training")
    del train_dataset
    del train_batches
    del valid_dataset
    del valid_batches
    del train_batch_lengths
    del valid_batch_lengths
    del train_data_loader
    del valid_data_loader

    # train_losses, valid_losses = train(train_data_loader, valid_data_loader, model, device, optim, criterion,
    #                                    config['num_epochs'], vocab)
    # logger.info(f'training losses is {train_losses[-1]}'
    #       f'validation losses is {valid_losses[-1]}')


def accuracy(outputs, targets, vocab):
    # define total accuracy, structure accuracy, control accuracy,
    # duration accuracy, pitch accuracy
    with torch.no_grad():
        # print('\n')
        accuracy_result = {}
        types_number_counter = {}
        all_type_token = set(vocab.token_class_ranges.values())
        for one_type_token in all_type_token:
            accuracy_result[one_type_token] = 0
            types_number_counter[one_type_token] = 0
        accuracy_result['total'] = 0
        types_number_counter['total'] = 0

        accuracy_result['track_control'] = 0
        types_number_counter['track_control'] = 0
        accuracy_result['bar_control'] = 0
        types_number_counter['bar_control'] = 0

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

                accuracy_result[target_classes] += token_idx.item() == target_idx
                types_number_counter[target_classes] += 1

                accuracy_result['total'] += token_idx.item() == target_idx
                types_number_counter['total'] += 1

        accuracy_result['track_control'] = accuracy_result['density'] + \
                                           accuracy_result['occupation'] + \
                                           accuracy_result['polyphony'] + \
                                           accuracy_result['pitch_register']

        types_number_counter['track_control'] = types_number_counter['density'] + \
                                                types_number_counter['occupation'] + \
                                                types_number_counter['polyphony'] + \
                                                types_number_counter['pitch_register']

        # accuracy_result['track_control'] /= types_number_counter['track_control']

        accuracy_result['bar_control'] = accuracy_result['tensile'] + \
                                         accuracy_result['diameter']

        types_number_counter['bar_control'] = types_number_counter['tensile'] + \
                                              types_number_counter['diameter']

        # accuracy_result['bar_control'] /= types_number_counter['bar_control']

        for token_type in accuracy_result.keys():
            if types_number_counter[token_type] != 0:
                accuracy_result[token_type] /= types_number_counter[token_type]

        return accuracy_result, generated_output, target_output


# def train(train_loader, valid_loader, model, device, optim, criterion, logger, num_epochs, vocab):
#     print_every = 100
#     model.train()
#
#     lowest_val = 1e9
#     train_losses = []
#     train_accuracies = {'total': [],
#                         'pitch': [],
#                         'duration': [],
#                         'structure': [],
#                         'time_signature': [],
#                         'tempo': [],
#                         'control': [],
#                         'program': [],
#                         'eos': []}
#     val_losses = []
#     total_step = 0
#     for epoch in range(num_epochs):
#         # pbar = tqdm(total=print_every, leave=False)
#         total_loss = 0
#         every_print_accuracy = {'total': 0,
#                                 'pitch': 0,
#                                 'duration': 0,
#                                 'structure': 0,
#                                 'tempo': 0,
#                                 'time_signature': 0,
#                                 'control': 0,
#                                 'program': 0,
#                                 'eos': 0}
#
#         # Shuffle batches every epoch
#         # train_loader.dataset.shuffle_batches()
#         for step, data in enumerate(iter(train_loader)):
#             total_step += 1
#
#             # Send the batches and key_padding_masks to gpu
#             src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
#             tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
#             tgt_out = data['target_out'].to(device)
#             memory_key_padding_mask = src_key_padding_mask.clone()
#
#             # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
#
#             tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to(device)
#
#             # Forward
#             optim.zero_grad()
#             outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
#             # logger.info(src.size())
#             # logger.info(tgt.size())
#             # logger.info("outside model, output_size:", outputs.size())
#
#             loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))
#
#             # Backpropagate and update optim
#             loss.backward()
#
#             # optim.step_and_update_lr()
#             optim.step()
#
#             total_loss += loss.item()
#             train_losses.append(loss.item())
#             # accuracies = accuracy(outputs,tgt_out,vocab)
#             # total_accuracy += accuracies['total']
#
#             # for key in train_accuracies.keys():
#             #     train_accuracies[key].append((step,accuracies[key]))
#
#             # logger.info(f'loss is {loss}')
#             # logger.info(f'total accuracy is {accuracies["total"]}')
#
#             # pbar.update(1)
#             if step % print_every == print_every - 1:
#                 # pbar.close()
#                 times = int((step / (print_every - 1)))
#
#                 src_token = []
#
#                 for i, output in enumerate(src[0]):
#                     output_token = vocab.index2char(output.item())
#                     src_token.append(output_token)
#
#                 accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)
#                 every_print_accuracy['total'] += accuracies['total']
#                 every_print_accuracy['pitch'] += accuracies['pitch']
#                 every_print_accuracy['duration'] += accuracies['duration']
#                 every_print_accuracy['structure'] += accuracies['structure']
#                 every_print_accuracy['tempo'] += accuracies['tempo']
#                 every_print_accuracy['time_signature'] += accuracies['time_signature']
#                 every_print_accuracy['program'] += accuracies['program']
#                 every_print_accuracy['eos'] += accuracies['eos']
#                 every_print_accuracy['control'] += accuracies['control']
#
#                 # lr = optim.get_lr()
#
#                 # logger.info(f'loss is {loss}')
#                 # logger.info(f'total accuracy is {accuracies["total"]}')
#                 logger.info(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \n '
#                             f'Train Loss: {total_loss / print_every} \t Accuracy {accuracies["total"] / times} \n'
#                             f'structure accuracy : {every_print_accuracy["structure"] / times} \n'
#                             f'duration accuracy : {every_print_accuracy["duration"] / times} \n'
#                             f'pitch accuracy : {every_print_accuracy["pitch"] / times} \n'
#                             f'program accuracy : {every_print_accuracy["program"] / times} \n'
#                             f'eos accuracy : {every_print_accuracy["eos"] / times} \n'
#                             f'tempo accuracy : {every_print_accuracy["tempo"] / times} \n'
#                             f'time signature accuracy : {every_print_accuracy["time_signature"] / times} \n'
#                             f'control accuracy : {every_print_accuracy["control"] / times} \n'
#                             f'input size is {src.size()} \n'
#                             f'input is : {src_token[:50]} \n'
#                             f'output size is {len(target_output)} \n'
#                             f'generated output: {generated_output[:50]} \n'
#                             f'target output: {target_output[:50]} \n'
#                             )
#
#                 total_loss = 0
#
#                 # pbar = tqdm(total=print_every, leave=False)
#
#         for key in train_accuracies.keys():
#             train_accuracies[key].append(every_print_accuracy[key] / times)
#
#         logger.info(f'Epoch [{epoch + 1} / {num_epochs} end] \t '
#                     f'ave train losses is {np.mean(train_losses)} \t')
#
#         for key in train_accuracies.keys():
#             logger.info(f' {key} accuracy is {train_accuracies[key][epoch]} \t ')
#
#         # Validate every epoch
#         # pbar.close()
#         val_loss, val_accuracy = validate(valid_loader, model, criterion, device, vocab, logger)
#         val_losses.append(val_loss)
#
#         with tune.checkpoint_dir(epoch) as checkpoint_dir:
#             path = os.path.join(checkpoint_dir, "checkpoint")
#             print(f'checkpoint_dir is {checkpoint_dir}')
#             torch.save((model.state_dict(), optim.state_dict()), path)
#
#         tune.report(loss=val_loss, accuracy=val_accuracy['total'])
#
#         # if val_loss < lowest_val:
#         #     lowest_val = val_loss
#         #     torch.save(model, 'output/transformer.pth')
#         # logger.info(f'Val Loss: {val_loss}, Val accuracy: {val_accuracy["total"]}')
#     return train_losses, val_losses


def validate(valid_loader, model,config, ce_loss,tempo_loss,density_loss,occupation_loss,polyphony_loss,pitch_register_loss,tensile_loss,diameter_loss, device, vocab, logger):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    total_loss = 0
    total_data_length = len(valid_loader)

    total_accuracy = {'total': 0,
                      'track_control': 0,
                      'bar_control': 0}

    for token_type in set(vocab.token_class_ranges.values()):
        total_accuracy[token_type] = 0

    # total_accuracy = {'total': 0,
    #                   'pitch': 0,
    #                   'duration': 0,
    #                   'structure': 0,
    #                   'tempo': 0,
    #                   'time_signature': 0,
    #                   'track_control': 0,
    #                   'bar_control': 0,
    #                   'density': 0,
    #                   'polyphony': 0,
    #                   'occupation': 0,
    #                   'pitch_register': 0,
    #                   'tensile': 0,
    #                   'diameter': 0,
    #                   'key': 0,
    #                   'program': 0,
    #                   'eos': 0}

    for data in iter(valid_loader):
        # Send the batches and key_padding_masks to gpu
        src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
        tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
        tgt_out = data['target_out'].to(device)
        memory_key_padding_mask = src_key_padding_mask.clone()

        # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)

        tgt_mask = gen_nopeek_mask(tgt_inp.shape[1])
        tgt_mask = torch.tensor(
            np.repeat(np.expand_dims(tgt_mask, 0), memory_key_padding_mask.shape[0], axis=0)).float()

        tgt_mask = tgt_mask.to(device)
        with torch.no_grad():
            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)

            loss_input_1 = rearrange(outputs, 'b t v -> (b t) v')
            loss_input_2 = rearrange(tgt_out, 'b o -> (b o)')

            loss1 = ce_loss(loss_input_1, loss_input_2)
            if config['loss_num'] == 1:
                loss2 = tempo_loss(loss_input_1, loss_input_2)
                loss3 = density_loss(loss_input_1, loss_input_2)
                loss4 = occupation_loss(loss_input_1, loss_input_2)
                loss5 = polyphony_loss(loss_input_1, loss_input_2)
                loss6 = pitch_register_loss(loss_input_1, loss_input_2)
                loss7 = tensile_loss(loss_input_1, loss_input_2)
                loss8 = diameter_loss(loss_input_1, loss_input_2)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + \
                       loss7 + loss8
            else:
                loss = loss1

            # if config['loss_num'] == 1:
            #     print(f'ce loss is {loss1}')
            #     print(f'tempo loss is {loss2}')
            #     print(f'density loss is {loss3}')
            #     print(f'occupation loss is {loss4}')
            #     print(f'polyphony loss is {loss5}')
            #     print(f'pitch_register loss is {loss6}')
            #     print(f'tensile loss is {loss7}')
            #     print(f'diameter loss is {loss8}')

            total_loss += loss.item()
            accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)

            for token_type in total_accuracy.keys():
                total_accuracy[token_type] += accuracies[token_type]
            # total_accuracy['total'] += accuracies['total']
            # total_accuracy['pitch'] += accuracies['pitch']
            # total_accuracy['duration'] += accuracies['duration']
            # total_accuracy['structure'] += accuracies['structure']
            # total_accuracy['tempo'] += accuracies['tempo']
            # total_accuracy['time_signature'] += accuracies['time_signature']
            # total_accuracy['program'] += accuracies['program']
            # total_accuracy['eos'] += accuracies['eos']
            # total_accuracy['track_control'] += accuracies['track_control']
            # total_accuracy['bar_control'] += accuracies['bar_control']
            # total_accuracy['density'] += accuracies['density']
            # total_accuracy['polyphony'] += accuracies['polyphony']
            # total_accuracy['occupation'] += accuracies['occupation']
            # total_accuracy['pitch_register'] += accuracies['pitch_register']
            # total_accuracy['tensile'] += accuracies['tensile']
            # total_accuracy['diameter'] += accuracies['diameter']
            # total_accuracy['key'] += accuracies['key']

            # lr = optim.get_lr()

            # logger.info(f'loss is {loss}')
            # logger.info(f'total accuracy is {accuracies["total"]}')

    total_loss /= total_data_length

    logger.info(f'validation loss is {total_loss}')
    for key in total_accuracy.keys():
        total_accuracy[key] /= total_data_length
        logger.info(f'validation {key} accuracy is {total_accuracy[key]}')

    src_token = []
    for i, output in enumerate(src[0]):
        output_token = vocab.index2char(output.item())
        src_token.append(output_token)

    logger.info(f'input size is {src.size()} \n'
                f'input is : {src_token[:50]} \n'
                f'output size is {len(target_output)} \n'
                f'generated output: {generated_output[:50]} \n'
                f'target output: {target_output[:50]} \n')

    # pbar = tqdm(total=print_every, leave=False)

    # pbar.update(1)

    # pbar.close()
    model.train()
    return total_loss, total_accuracy


def test_loss_accuracy(test_loader, model, criterion, device, vocab, logger):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    total_loss = 0
    total_data_length = len(test_loader)
    # total_accuracy = {'total': 0,
    #                   'pitch': 0,
    #                   'duration': 0,
    #                   'structure': 0,
    #                   'tempo': 0,
    #                   'time_signature': 0,
    #                   'track_control': 0,
    #                   'bar_control': 0,
    #                   'density': 0,
    #                   'polyphony': 0,
    #                   'occupation': 0,
    #                   'pitch_register': 0,
    #                   'tensile': 0,
    #                   'diameter': 0,
    #                   'key': 0,
    #                   'program': 0,
    #                   'eos': 0}

    total_accuracy = {'total': 0,
                      'track_control': 0,
                      'bar_control': 0}

    for token_type in set(vocab.token_class_ranges.values()):
        total_accuracy[token_type] = 0

    for data in iter(test_loader):
        # Send the batches and key_padding_masks to gpu
        src, src_key_padding_mask = data['input'].to(device), data['input_pad_mask'].to(device)
        tgt_inp, tgt_key_padding_mask = data['target_in'].to(device), data['target_pad_mask'].to(device)
        tgt_out = data['target_out'].to(device)
        memory_key_padding_mask = src_key_padding_mask.clone()

        # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)

        tgt_mask = gen_nopeek_mask(tgt_inp.shape[1])
        tgt_mask = torch.tensor(
            np.repeat(np.expand_dims(tgt_mask, 0), memory_key_padding_mask.shape[0], axis=0)).float()

        tgt_mask = tgt_mask.to(device)
        with torch.no_grad():
            outputs = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
            loss = criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))

            total_loss += loss.item()
            accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)
            # total_accuracy['total'] += accuracies['total']
            # total_accuracy['pitch'] += accuracies['pitch']
            # total_accuracy['duration'] += accuracies['duration']
            # total_accuracy['structure'] += accuracies['structure']
            # total_accuracy['tempo'] += accuracies['tempo']
            # total_accuracy['time_signature'] += accuracies['time_signature']
            # total_accuracy['program'] += accuracies['program']
            # total_accuracy['eos'] += accuracies['eos']
            # total_accuracy['track_control'] += accuracies['track_control']
            # total_accuracy['bar_control'] += accuracies['bar_control']
            # total_accuracy['density'] += accuracies['density']
            # total_accuracy['polyphony'] += accuracies['polyphony']
            # total_accuracy['occupation'] += accuracies['occupation']
            # total_accuracy['pitch_register'] += accuracies['pitch_register']
            # total_accuracy['tensile'] += accuracies['tensile']
            # total_accuracy['diameter'] += accuracies['diameter']
            # total_accuracy['key'] += accuracies['key']
            for token_type in total_accuracy.keys():
                total_accuracy[token_type] += accuracies[token_type]

            # lr = optim.get_lr()

            # logger.info(f'loss is {loss}')
            # logger.info(f'total accuracy is {accuracies["total"]}')

    total_loss /= total_data_length

    logger.info(f'test loss is {total_loss}')
    for key in total_accuracy.keys():
        total_accuracy[key] /= total_data_length
        logger.info(f'test {key} accuracy is {total_accuracy[key]}')

    src_token = []
    for i, output in enumerate(src[0]):
        output_token = vocab.index2char(output.item())
        src_token.append(output_token)

    logger.info(f'input size is {src.size()} \n'
                f'input is : {src_token[:50]} \n'
                f'output size is {len(target_output)} \n'
                f'generated output: {generated_output[:50]} \n'
                f'target output: {target_output[:50]} \n')

    # pbar = tqdm(total=print_every, leave=False)

    # pbar.update(1)

    # pbar.close()

    return total_loss, total_accuracy


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

from pathlib import Path
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import preprocessing
from preprocessing import event_2midi
import tension_calculation
import dataset
import pretty_midi
import os
from vocab import *
from generation import sampling,\
    weighted_sampling,nucleus,\
    softmax_with_temperature,total_duration,\
    clear_pitch_duration_event,cal_duration,\
    cal_track_control
import math
import logging
import coloredlogs
import pickle

from torch.optim import Adam

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ParallelLanguageDataset
from dataset import WordVocab
from model import ScoreTransformer
from dataset import all_tokens
from dataset import collate_mlm
from torch.nn import functional as F

import wandb
from log import logger
wandb.login()


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
    parser.add_argument('-v', '--device', default='', type=str,
                        help="device")
    parser.add_argument('-l', '--ce_only', default=False, type=bool,
                        help="ce only loss or with ordinal loss")
    parser.add_argument('-s', '--distance', default='medium', type=str,
                        help="ordinal loss distance")
    parser.add_argument('-c', '--checkpoint_dir', default="", type=str,
                        help="checkpoint dir")
    parser.add_argument('-r', '--learning_rate', default="0.0001", type=float,
                        help="learning rate")
    parser.add_argument('-w', '--control_token_weight', default=1, type=float,
                        help="control token weight")
    parser.add_argument('-i', '--run_id', default=None, type=str,
                        help="run id")
    parser.add_argument('-a', '--reset_lr', default=False, type=bool,
                        help="if to reset lr to 0.0001 after loading the checkpoint")

    parser.add_argument('-n', '--new_data', default=False, type=bool,
                        help="use generated data to train, default false")
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


def phi(rt, ri,distance='medium'):
    rt = rt.type(torch.float32)
    if distance == 'small':
        return torch.abs(rt - ri)
    elif distance == 'large':
        return 2*torch.square(rt - ri)
    else:
        return torch.square(rt - ri)


def soft_label(total_label_num, target_index_range,distance):
    target_index_length = target_index_range[1] - target_index_range[0] + 1
    output_weights = torch.zeros(total_label_num, total_label_num)
    weights = nn.functional.softmax(-phi(torch.unsqueeze(torch.arange(target_index_length), dim=1),
                                         torch.arange(target_index_length),distance), 0)
    output_weights[target_index_range[0]:target_index_range[1] + 1,
    target_index_range[0]:target_index_range[1] + 1] = weights
    return output_weights


class OrdinalLoss(nn.Module):
    def __init__(self, target_index_range, vocab_size,distance, device, **kwargs):
        super().__init__()
        # a tuple to delineate the target index
        self.weights = soft_label(vocab_size, target_index_range,distance).to(device)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x, target):
        logsoftmax = self.logsoftmax(x)
        target_array = -self.weights[target]
        return torch.mean(torch.sum(torch.multiply(logsoftmax, target_array), axis=1))


def main(**kwargs):
    # device = torch.device(kwargs['device'])
    # print(f'device is {device}')
    args = get_args()

    global platform, num_epochs, is_debug

    # event_folder = args.event_folder
    # tension_folder = args.tension_folder
    # train_ratio = args.train_ratio
    # valid_ratio = args.valid_ratio
    # test_ratio = args.test_ratio
    num_epochs = args.num_epochs
    platform = args.platform
    is_debug = args.is_debug
    checkpoint_dir = args.checkpoint_dir
    distance = args.distance
    lr = args.learning_rate
    run_id = args.run_id
    is_ce_only = args.ce_only
    control_token_weight = args.control_token_weight
    use_new_data = args.new_data
    #device = args.device

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
    print(f'is_debug is {is_debug}')
    print(f'platform is {platform}')
    if run_id:
        print(f'run_id is {run_id}')
    if args.reset_lr:
        print(f'reset lr to 0.0001')
    print(f'is_ce_only is {is_ce_only}')
    if not is_ce_only:
        print(f'ordinal distance is {distance}')
    print(f'learning rate is {lr}')
    print(f'control token weight is {control_token_weight}')
    print(f'fine tuning')
    print(f'use new data is {use_new_data}')
    if checkpoint_dir:
        print(f'checkpoint dir is {checkpoint_dir}')

    # print(f'max_token_length is {max_token_length}')
    # print(f'train jointly is {train_jointly}')

    config = {"batch_size": 2,

              "span_lengths": 3,
              "span_ratio_jointly": 0.5,
              "eos_weight": 1,
              # "train_jointly": tune.grid_search([True]),
              'd_model': 512,
              'lr':lr,
              'lr_reset':args.reset_lr,
              'num_encoder_layers': 4,
              'ce_loss_only': is_ce_only,
              'distance': distance,
              'epochs':num_epochs,
              'use_new_data':use_new_data,
              'control_token_weight':control_token_weight,
              # 'num_decoder_layers': tune.grid_search([4]),
              # 'dim_feedforward': tune.grid_search([2048]),
              'nhead': 8,
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
    #
    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=num_epochs,
    #     grace_period=8,
    #     reduction_factor=2)
    # reporter = CLIReporter(
    #     # parameter_columns=["l1", "l2", "lr", "batch_size"],
    #     metric_columns=[
    #         "train_loss",
    #         "train_accuracy",
    #         "lr",
    #         "loss",
    #         "val_loss",
    #         "val_accuracy",
    #         "pitch_accuracy",
    #         "duration_accuracy",
    #         "structure_accuracy",
    #         "tempo_accuracy",
    #         "time_signature_accuracy",
    #         "program_accuracy",
    #         "eos_accuracy",
    #         "track_control_accuracy",
    #         "bar_control_accuracy",
    #         "density_accuracy",
    #         "occupation_accuracy",
    #         "polyphony_accuracy",
    #         "pitch_register_accuracy",
    #         "tensile_accuracy",
    #         "diameter_accuracy",
    #         "key_accuracy"])
    run(config,checkpoint_dir,run_id)



    # if is_debug:
    #     train_ray_tune(config,checkpoint_dir=checkpoint_dir)
    # else:
    #     result = tune.run(
    #         train_ray_tune,
    #         resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    #         config=config,
    #         num_samples=num_samples,
    #         scheduler=scheduler,
    #         progress_reporter=reporter,
    #         local_dir='./ray_result/'
    #     )

        # best_trial = result.get_best_trial("loss", "min", "last")
        # print("Best trial config: {}".format(best_trial.config))
        # print("Best trial final validation loss: {}".format(
        #     best_trial.last_result["loss"]))
        # print("Best trial final validation accuracy: {}".format(
        #     best_trial.last_result["total"]))

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
        # test_batches = pickle.load(open(folder_prefix + 'score_transformer/sync/test_batches_0_0_8', 'rb'))
        # test_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/sync/test_batch_lengths_0_0_8', 'rb'))
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

import random
import re
import copy


def cal_bar_control(generated_name):
    result = tension_calculation.extract_notes(pretty_midi.PrettyMIDI(generated_name), 3)

    if result is None:
        return None
    pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result

    key_name = tension_calculation.all_key_names

    result_generated = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices, -1, key_name)

    total_tension_generated, diameters_generated, key_name_generated = result_generated

    generated_tensile_category = dataset.to_category(total_tension_generated, dataset.tensile_bins)
    generated_diameter_category = dataset.to_category(diameters_generated, dataset.tensile_bins)


    tensile_string = [f's_{str(tensile)}' for tensile in generated_tensile_category]
    diameter_string = [f'a_{str(diameter)}' for diameter in generated_diameter_category]
    return tensile_string, diameter_string,key_name_generated



def model_generate(model, src, tgt,device,return_weights=False):

    src = src.clone().detach().unsqueeze(0).long().to(device)
    tgt = torch.tensor(tgt).clone().detach().unsqueeze(0).to(device)
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    tgt_mask = tgt_mask.clone().detach().unsqueeze(0).to(device)


    output,weights = model(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                           tgt_mask=tgt_mask)
    if return_weights:
        return output.squeeze(0).to('cpu'), weights.squeeze(0).to('cpu')
    else:
        return output.squeeze(0).to('cpu')




def prediction(model,event,device, vocab):
    all_meta_pos = []
    tokens = []
    target_outputs = []
    generate_outputs = []
    for pos, token in enumerate(event):
        if token in all_meta_tokens:
            all_meta_pos.append(pos)
    # tokens to index
    start_pos = 0
    while start_pos < len(event):
        tokens.append(vocab.char2index(event[start_pos]))
        start_pos += 1

    for pos_chosen in all_meta_pos:
        src = copy.copy(tokens)
        masked_token = event[pos_chosen]
        src[pos_chosen] = vocab.mask_indices[0]

        target_output = masked_token

        target_outputs.append(target_output)

        # logger.info(f'target output is: {target_output}')

        tgt_inp = []

        for src_pos_chosen in all_meta_pos:
            if src_pos_chosen != pos_chosen:
                src[src_pos_chosen] = vocab.ignore_indices[0]

        with torch.no_grad():
            sampling_times = 0
            tgt_inp.append(vocab.mask_indices[0])
            output = model_generate(model, torch.tensor(src).long(), tgt_inp,device)
            index = sampling(output[-1], 0.9)
            output_token = vocab.index2char(index)

            while vocab.token_class_ranges[index] != vocab.token_class_ranges[vocab.char2index(target_output)] and sampling_times < 10:
                index = sampling(output[-1], 0.9)
                output_token = vocab.index2char(index)
                sampling_times += 1

            generate_outputs.append(output_token)

    return generate_outputs,target_outputs




def mask_bar_and_track(event,vocab,mode):

    mask_mode = mode
    tokens = []

    decoder_target = []
    masked_indices_pairs = []
    mask_bar_names = []
    mask_track_names = []
    bar_poses = np.where(np.array(event) == 'bar')[0]

    r = re.compile('i_\d')

    track_program = list(filter(r.match, event))
    track_nums = len(track_program)
    track_end_poses = []
    if track_nums == 3:
        track_0_pos = np.where('track_0' == np.array(event))[0]
        track_1_pos = np.where('track_1' == np.array(event))[0]
        track_2_pos = np.where('track_2' == np.array(event))[0]
        for pos in track_2_pos[:-1]:
            track_end_poses.append(bar_poses[np.where(pos < bar_poses)[0][0]])
        else:
            track_end_poses.append(len(event))
        all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos, track_2_pos, track_end_poses]))

    elif track_nums == 2:
        #         ratios = track_ratio[1]
        track_0_pos = np.where('track_0' == np.array(event))[0]
        track_1_pos = np.where('track_1' == np.array(event))[0]
        for pos in track_1_pos[:-1]:
            track_end_poses.append(bar_poses[np.where(pos < bar_poses)[0][0]])
        else:
            track_end_poses.append(len(event))
        all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos, track_end_poses]))

    else:
        track_0_pos = np.where('track_0' == np.array(event))[0]
        for pos in track_0_pos[:-1]:
            track_end_poses.append(bar_poses[np.where(pos < bar_poses)[0][0]])
        else:
            track_end_poses.append(len(event))
        all_track_pos = np.sort(np.concatenate([track_0_pos, track_end_poses]))

    bar_with_track_poses = []

    for i, pos in enumerate(all_track_pos):
        if i % (track_nums + 1) == 0:
            this_bar_poses = []
            this_bar_pairs = []
            this_bar_poses.append(pos)

        else:
            this_bar_poses.append(pos)
            if i % (track_nums + 1) == track_nums:
                for j in range(len(this_bar_poses) - 1):
                    this_bar_pairs.append((this_bar_poses[j], this_bar_poses[j + 1]))

                bar_with_track_poses.append(this_bar_pairs)

    # 25% mask whole tracks(select from 1 to track_num track)
    # 25% mask whole bars(select from 1 to bar num)
    # 50% mask random bars(select from 1 to bar num, random tracks
    # (select 1 from track number in a bar)

    if mask_mode == 0:
        bar_mask_number = np.random.randint(0,len(bar_poses))
        bar_mask_poses = np.sort(np.random.choice(len(bar_poses),size=bar_mask_number+1,replace=False))

        for bar_mask_pos in bar_mask_poses:
            track_mask_number = np.random.randint(0, track_nums)
            track_mask_poses = np.sort(np.random.choice(track_nums,size=track_mask_number+1,replace=False))
            for track_mask_pos in track_mask_poses:
                mask_track_names.append(track_mask_pos)
                mask_bar_names.append(bar_mask_pos)
                bar_with_track_poses[bar_mask_pos][track_mask_pos]
                masked_indices_pairs.append(bar_with_track_poses[bar_mask_pos][track_mask_pos])
    elif mask_mode == 1:
        # mask whole tracks
        track_mask_number = np.random.randint(0, track_nums)
        track_mask_poses = np.sort(np.random.choice(track_nums, size=track_mask_number + 1, replace=False))

        for bar_num, tracks_in_a_bar in enumerate(bar_with_track_poses):
            for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                if track_pos in track_mask_poses:
                    mask_bar_names.append(bar_num)
                    mask_track_names.append(track_pos)
                    masked_indices_pairs.append(track_star_end_poses)

    else:
        # mask whole bars
        bar_mask_number = np.random.randint(0, len(bar_poses))
        bar_mask_poses = np.sort(np.random.choice(len(bar_poses),size=bar_mask_number+1,replace=False))

        for bar_mask_pos in bar_mask_poses:
            for tracks_in_a_bar in bar_with_track_poses[bar_mask_pos]:
                for track_name in range(track_nums):
                    mask_bar_names.append(bar_mask_pos)
                    mask_track_names.append(track_name)
                masked_indices_pairs.append(tracks_in_a_bar)

    assert len(mask_bar_names) == len(mask_track_names)

    token_events = event.copy()

    for masked_pairs in masked_indices_pairs:
        masked_token = event[masked_pairs[0]:masked_pairs[1]]

        for token in masked_token:
            decoder_target.append(vocab.char2index(token))
        else:
            decoder_target.append(vocab.eos_index)

    for masked_pairs in masked_indices_pairs[::-1]:
        # print(masked_pairs)
        # print(token_events[masked_pairs[0]:masked_pairs[1]])
        for pop_time in range(masked_pairs[1] - masked_pairs[0]):
            token_events.pop(masked_pairs[0])
        token_events.insert(masked_pairs[0], 'm_0')

    for token in token_events:
        tokens.append(vocab.char2index(token))

    tokens = np.array(tokens)
    decoder_target = np.array(decoder_target)

    return tokens, decoder_target, mask_track_names, mask_bar_names

def cut_duration(event):
    return event
def generation(model,event,device, vocab,mode,temperature=0.9):
    src, tgt_out, mask_track_names, mask_bar_names = mask_bar_and_track(event, vocab,  mode)

    duration_name_to_time, duration_time_to_name, duration_times, bar_duration = preprocessing.get_note_duration_dict(
        1, (int(event[0][0]), int(event[0][2])))

    src_masked_track = np.sum(src == vocab.char2index('m_0'))
    tgt_inp = []
    generate_times = 0
    weight_list = []
    tgt_inp_list = []

    with torch.no_grad():
        mask_idx = 0
        while mask_idx < src_masked_track:
            mask_track_name = 'track_' + f'{str(mask_track_names[mask_idx])}'
            # logger.info(f'current mask idx is {mask_idx}')

            # for mask_idx in range(src_masked_track):
            this_tgt_inp = []

            this_tgt_failure = False
            track_name_failure = False
            pitch_failure = False
            duration_failure = False

            this_tgt_inp.append(vocab.char2index('m_0'))

            curr_time = 0
            previous_duration = 0

            in_duration_event = False
            is_rest_s = False

            in_pitch_event = False

            duration_list = []

            while this_tgt_inp[-1] != vocab.char2index('<eos>') and len(this_tgt_inp) < 500:
                sampling_times = 0
                output, weight = model_generate(model, torch.tensor(src), tgt_inp + this_tgt_inp, device, return_weights=True)
                index = sampling(output[-1], 1.2)
                sampling_times += 1

                event = vocab.index2char(index)
                # logger.info(event)

                if len(this_tgt_inp) == 1:

                    compare_event = event
                    if compare_event != mask_track_name:
                        while compare_event != mask_track_name and sampling_times < 10:
                            index = sampling(output[-1], temperature)
                            event = vocab.index2char(index)
                            sampling_times += 1

                            compare_event = event
                        if compare_event != mask_track_name:
                            sampling_times = 0
                            while compare_event != mask_track_name and sampling_times < 10:
                                output, weight = model_generate(model, torch.tensor(src), tgt_inp + this_tgt_inp, device,
                                                                return_weights=True)
                                index = sampling(output[-1], temperature)
                                event = vocab.index2char(index)
                                sampling_times += 1

                                compare_event = event
                            if compare_event != mask_track_name:
                                # logger.info(f'{compare_event} is not equal to {mask_track_name}')
                                # logger.info(f'mask {mask_idx} needs to be generated again')
                                this_tgt_failure = True
                                track_name_failure = True

                if event in pitches:
                    in_pitch_event = True
                    if in_duration_event:
                        curr_time, previous_duration = clear_pitch_duration_event(
                            curr_time,
                            previous_duration,
                            is_rest_s,
                            duration_list, duration_name_to_time)

                        duration_list = []

                        in_duration_event = False
                        is_rest_s = False

                if event in duration_name_to_time.keys():
                    if not in_duration_event and not in_pitch_event:
                        # generate a pitch event token
                        while event not in pitches and sampling_times < 10:
                            index = sampling(output[-1], temperature)
                            event = vocab.index2char(index)
                            sampling_times += 1
                        if event not in pitches:
                            sampling_times = 0
                            while event not in pitches and sampling_times < 10:
                                output, weight = model_generate(model, torch.tensor(src), tgt_inp + this_tgt_inp, device,
                                                                return_weights=True)
                                index = sampling(output[-1], temperature)
                                event = vocab.index2char(index)
                                sampling_times += 1
                            if event != pitches:
                                # logger.info(f'{event} is not pitch')
                                # logger.info(f'mask {mask_idx} needs to be generated again')
                                this_tgt_failure = True
                                pitch_failure = True

                        in_pitch_event = True
                    else:
                        in_pitch_event = False
                        duration_list.append(event)
                        in_duration_event = True

                # an event not in duration event happens

                if event == 'rest_s':
                    is_rest_s = True

                if event == '<eos>':
                    if in_duration_event:
                        curr_time, previous_duration = clear_pitch_duration_event(
                            curr_time,
                            previous_duration,
                            is_rest_s,
                            duration_list, duration_name_to_time)


                    if not math.isclose(curr_time, bar_duration):
                        # logger.info(f'{curr_time} is not equal to {bar_duration}')
                        # logger.info(f'mask {mask_idx} needs to be generated again')
                        this_tgt_failure = True
                        duration_failure = True

                this_tgt_inp.append(index)

            if this_tgt_inp[-1] == vocab.char2index('<eos>') and not this_tgt_failure:
                mask_idx += 1
                tgt_inp.extend(this_tgt_inp[:-1])
                weight_list.append(weight)
                this_tgt_inp_tokens = []
                for index in this_tgt_inp[:-1]:
                    this_tgt_inp_tokens.append(vocab.index2char(index))
                tgt_inp_list.append(this_tgt_inp_tokens)

            # no eos after long generation
            elif this_tgt_inp[-1] != vocab.char2index('<eos>'):
                return None
            else:
                # logger.info(f'generate again, generate time is {generate_times}')
                generate_times += 1
                if generate_times > 2:
                    # logger.info(f'fail to have correct track duration {mask_idx} after 10 times')
                    #return None
                    if track_name_failure:
                        # logger.info('manually correct track name')
                        tgt_inp.append(vocab.char2index(mask_track_name))
                    if duration_failure:
                        # logger.info('duration failure')
                        # logger.info('manually correct duration')
                        this_tgt_inp = cut_duration(this_tgt_inp)
                        tgt_inp.extend(this_tgt_inp[:-1])
                    if pitch_failure:
                        # logger.info('pitch failure')
                        return None

                    weight_list.append(None)

                    mask_idx += 1

    generated_output = []
    target_output = []
    src_token = []

    for i, token_idx in enumerate(tgt_inp):
        output_token = vocab.index2char(token_idx)
        generated_output.append(output_token)

    for i, token_idx in enumerate(tgt_out):
        target_token = vocab.index2char(token_idx.item())
        target_output.append(target_token)

    # logger.info(f'generation {generated_output}')
    # logger.info(f'target {target_output}')

    for i, token_idx in enumerate(src):
        src_token.append(vocab.index2char(token_idx.item()))


    return restore_marked_input(src_token, generated_output), mask_track_names,mask_bar_names




def restore_marked_input(src_token, generated_output):
    src_token = np.array(src_token,dtype='<U9')

    # restore with generated output
    restored_with_generated_token = src_token.copy()

    generated_output = np.array(generated_output)

    generation_mask_indices = np.where(generated_output == 'm_0')[0]

    if len(generation_mask_indices) == 1:

        mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
        generated_result_sec = generated_output[generation_mask_indices[0] + 1:]

        #         logger.info(len(generated_result_sec))
        restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
        for token in generated_result_sec[::-1]:
            #             logger.info(token)
            restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)


    else:

        for i in range(len(generation_mask_indices) - 1):
            #         logger.info(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i] + 1:generation_mask_indices[i + 1]]

            #             logger.info(len(generated_result_sec))
            #             logger.info(mask_indices[i])
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])

            for token in generated_result_sec[::-1]:
                #                 logger.info(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)

        else:
            #         logger.info(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i + 1] + 1:]

            #             logger.info(len(generated_result_sec))
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
            for token in generated_result_sec[::-1]:
                #                 logger.info(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)


    return restored_with_generated_token

def bar_track_validate(output_indices, index, mask_track_name, time_signature,total_duration,vocab):
    if len(output_indices) == 1:
        # index must match track name
        if vocab.char2index(index) != mask_track_name:
            return False
    if vocab.char2index(index) == '<eos>':
        duration = cal_duration(output_indices,time_signature)
        if total_duration != duration:
            return False
    return True




def generate_and_prediction(model,event,device,vocab):
    # first predict all the control tokens
    logger.info('generation/prediction test')
    generated_outputs,target_outputs = prediction(model,event,device,vocab)


    logger.info(f'predicted output is {generated_outputs[:40]}')
    logger.info(f'target output is {target_outputs[:40]}')
    prediction_accuracy = generation_accuracy(generated_outputs,target_outputs,vocab)
    logger.info(f'prediction accuracy is {prediction_accuracy}')

    accuracy_result = {}
    accuracy_result['bar_control'] = []
    accuracy_result['track_control'] = []
    accuracy_result['tensile'] = []
    accuracy_result['diameter'] = []
    accuracy_result['density'] = []
    accuracy_result['polyphony'] = []
    accuracy_result['occupation'] = []
    accuracy_result['pitch_register'] = []

    # generation
    bar_poses = np.where(np.array(event) == 'bar')[0]

    r = re.compile('i_\d')

    track_program = list(filter(r.match, event))
    track_nums = len(track_program)

    target_track_control = event[3:bar_poses[0]-track_nums]
    r = re.compile('(?:a_|s_)')

    target_bar_control = list(filter(r.match, event))
    # 1. random mask bar and track
    # 2. mask one/several tracks

    for mode in range(2):
        tensile_accuracy = 0
        diameter_accuracy = 0
        bar_control_accuacy = 0

        track_control_accuracy = 0
        density_control_accuracy = 0
        polyphony_control_accuracy = 0
        occupation_control_accuracy = 0
        pitch_register_control_accuracy = 0

        result = generation(model, event, device, vocab, mode)
        if result is None:
            return None
        else:
            generated_tokens, mask_track_names, mask_bar_names = result

        # logger.info(f'mask_track_names is {mask_track_names}')
        # logger.info(f'mask_bar_names is {mask_bar_names}')
        generated_pm, _ = event_2midi(generated_tokens.tolist())
        generated_track_control = cal_track_control(generated_tokens,
                                                    generated_pm)
        generated_pm.write('./temp.mid')
        bar_controls = cal_bar_control('./temp.mid')
        if bar_controls is not None:
            generated_tensile, generated_diameter,key_name = bar_controls

        if len(generated_tensile) != len(bar_poses):
            bar_controls = None
            logger.info('generated bar is not equal to original bar length')

        if mode == 0:
            mask_track_bar_names = []
            logger.info(f'random mask bars and tracks')
            for i in range(len(mask_track_names)):
                mask_track_bar_names.append((mask_bar_names[i],mask_track_names[i]))
            logger.info(f'random mask bar/track name {mask_track_bar_names} \n')
        else:
            logger.info(f'mask whole track {np.unique(np.array(mask_track_names))}')

        mask_track_num = len(np.unique(np.array(mask_track_names)))
        mask_bar_num = len(np.unique(np.array(mask_bar_names)))

        if generated_track_control is not None:
            generated_track_control_in_tracks = []
            target_track_control_in_tracks = []

            for i, control in enumerate(generated_track_control):
                if i % track_nums in mask_track_names:
                    generated_track_control_in_tracks.append(generated_track_control[i])
                    target_track_control_in_tracks.append(target_track_control[i])

                    if vocab.token_class_ranges[vocab.char2index(control)] == 'density':
                        density_control_accuracy += control == target_track_control[i]
                        track_control_accuracy += control == target_track_control[i]
                    elif vocab.token_class_ranges[vocab.char2index(control)] == 'polyphony':
                        polyphony_control_accuracy += control == target_track_control[i]
                        track_control_accuracy += control == target_track_control[i]
                    elif vocab.token_class_ranges[vocab.char2index(control)] == 'occupation':
                        occupation_control_accuracy += control == target_track_control[i]
                        track_control_accuracy += control == target_track_control[i]
                    else:
                        pitch_register_control_accuracy += control == target_track_control[i]
                        track_control_accuracy += control == target_track_control[i]

            logger.info(f'generated track control is {generated_track_control_in_tracks}\n'
                        f'target track control is {target_track_control_in_tracks}')

            accuracy_result['track_control'].append(track_control_accuracy / mask_track_num / 4)
            accuracy_result['density'].append(density_control_accuracy / mask_track_num)
            accuracy_result['polyphony'].append(polyphony_control_accuracy / mask_track_num)
            accuracy_result['occupation'].append(occupation_control_accuracy / mask_track_num)
            accuracy_result['pitch_register'].append(pitch_register_control_accuracy / mask_track_num)
        else:
            logger.info('track control is None, skip')

        bar_control_in_bars = []
        target_bar_control_in_bars = []

        if bar_controls is not None:
            for bar_number in np.unique(np.array(mask_bar_names)):
                bar_control_in_bars.append(generated_tensile[bar_number])
                bar_control_in_bars.append(generated_diameter[bar_number])
                target_bar_control_in_bars.extend(target_bar_control[bar_number*2:bar_number*2+2])

                tensile_accuracy += generated_tensile[bar_number] == target_bar_control[bar_number*2]
                bar_control_accuacy += generated_tensile[bar_number] == target_bar_control[bar_number*2]
                diameter_accuracy += generated_diameter[bar_number] == target_bar_control[bar_number*2+1]
                bar_control_accuacy += generated_diameter[bar_number] == target_bar_control[bar_number*2+1]
            accuracy_result['bar_control'].append(bar_control_accuacy / mask_bar_num)
            accuracy_result['tensile'].append(tensile_accuracy / mask_bar_num)
            accuracy_result['diameter'].append(diameter_accuracy / mask_bar_num)

            logger.info(
                f'generated bar control is  {bar_control_in_bars} \n'
                f'target bar control is {target_bar_control_in_bars}')


        else:
            logger.info('bar control is None')

def run(config, checkpoint_dir=None,run_id=None):
    if is_debug:
        mode = 'offline'
    else:
        mode='online'

    if run_id:
        resume = 'allow'
    else:
        resume = None

    if platform == 'local':
        wandb_dir = '/home/data/guorui'
    else:
        wandb_dir = './'
    with wandb.init(project="score_transformer",mode=mode, dir=wandb_dir, tags=['finetune'],config=config,resume=resume,id=run_id):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        current_folder = wandb.run.dir
        logfile = current_folder + '/logging.log'
        if resume:
            filemode = 'a'
        else:
            filemode = 'w'
        logger.handlers = []
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                            datefmt='%Y-%m-%d %H:%M:%S', filename=logfile, filemode=filemode)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logger.addHandler(console)

        coloredlogs.install(level='INFO', logger=logger, isatty=False)

        vocab = WordVocab(all_tokens)

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


        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        optim = Adam(model.parameters(), lr=config['lr'])

        if checkpoint_dir:

            logger.info(f'load checkpoint from{checkpoint_dir}')
            model_dict = torch.load(checkpoint_dir)

            model_state = model_dict['model_state_dict']
            optimizer_state = model_dict['optimizer_state_dict']
            # start_epoch = model_dict['epoch'] + 1
            start_epoch = 0
            # logger.info(f'continue previous run from epoch {start_epoch}')


            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            #
            # new_state_dict = model_state

            model.load_state_dict(new_state_dict)

        else:
            start_epoch = 0


        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        model.to(device)

        if checkpoint_dir:

            optim.load_state_dict(optimizer_state)
            print(f'optim loaded lr is {optim.param_groups[0]["lr"]}')
            if config['lr_reset']:
                optim.param_groups[0]["lr"] = 0.0001
                print(f'reset lr to {optim.param_groups[0]["lr"]}')





        window_size = int(16 / 2)

        if platform == 'local':
            folder_prefix = '/home/ruiguo/'
        else:
            folder_prefix = '/content/drive/MyDrive/'

        if config['use_new_data']:
            train_batch_name = 'all_batches_new'
            train_length_name = 'batch_length_new'
            if is_debug:
                valid_batch_name = 'valid_batches_0_0_1'
                valid_batch_length_name = 'valid_batch_lengths_0_0_1'
            else:
                valid_batch_name = 'valid_batches_0_0_8'
                valid_batch_length_name = 'valid_batch_lengths_0_0_8'
        else:

            if is_debug:
                train_batch_name = 'all_batches_0'
                train_length_name = 'batch_length_0'
                valid_batch_name = 'valid_batches_new'
                valid_batch_length_name = 'valid_batch_lengths_new'
                test_batch_name = 'test_batches_new'

            else:
                train_batch_name = 'all_batches_1'
                train_length_name = 'batch_length_1'
                valid_batch_name = 'valid_batches_new'
                valid_batch_length_name = 'valid_batch_lengths_new'
                test_batch_name = 'test_batches_new'

        train_batches = pickle.load(open(folder_prefix + 'score_transformer/sync/' + train_batch_name, 'rb'))
        # train_batches = train_batches[18000:]
        train_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/sync/' + train_length_name, 'rb'))


        valid_batches = pickle.load(open(folder_prefix + 'score_transformer/sync/' + valid_batch_name, 'rb'))

        valid_batch_lengths = pickle.load(
            open(folder_prefix + 'score_transformer/sync/' + valid_batch_length_name, 'rb'))

        test_batches = pickle.load(open(folder_prefix + 'score_transformer/sync/' + test_batch_name, 'rb'))
        # test_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/sync/test_batch_lengths_0_0_1', 'rb'))

        logger.info(f'train batches file is  {folder_prefix + "score_transformer/" + train_batch_name}')
        logger.info(f'valid batches file is  {folder_prefix + "score_transformer/" + valid_batch_name}')

        logger.info(f'train batch length is {len(train_batches)}')
        logger.info(f'valid batch length is {len(valid_batches)}')
        logger.info(f'test batch length is {len(test_batches)}')



        train_dataset = ParallelLanguageDataset('',
                                                '',
                                                vocab,
                                                0, 0,
                                                2200,
                                                window_size,
                                                train_batches,
                                                train_batch_lengths,
                                                config['batch_size'],
                                                .15,
                                                .3,
                                                .3,
                                                .3,
                                                control_mask_ratio=.9,
                                                header_mask_ratio=.3,
                                                ignore_ratio=0.05,
                                                span_lengths=3,
                                                span_ratio_jointly=config['span_ratio_jointly'],
                                                span_ratio_separately_each_epoch=span_ratio_separately_each_epoch,
                                                logger=logger,
                                                mask_bar_num_ratio=None,
                                                mask_track_num_ratio=None,
                                                mask_bar_ctrl_token=False,
                                                pretraining=False,
                                                fine_tune_prediction=False,
                                                train_jointly=True)

        valid_dataset = ParallelLanguageDataset('',
                                                '',
                                                vocab, 0,
                                                0,
                                                2200,
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
                                                span_ratio_jointly=config['span_ratio_jointly'],
                                                span_ratio_separately_each_epoch=span_ratio_separately_each_epoch,
                                                logger=logger,
                                                mask_bar_num_ratio=None,
                                                mask_track_num_ratio=None,
                                                mask_bar_ctrl_token=False,
                                                pretraining=False,
                                                fine_tune_prediction=False,
                                                train_jointly=True)


        # test_dataset = ParallelLanguageDataset('', '',
        #                                        vocab, 0,
        #                                        0,
        #                                        2400,
        #                                        window_size,
        #                                        test_batches,
        #                                        test_batch_lengths,
        #                                        config['batch_size'],
        #                                        .15,
        #                                        .3,
        #                                        .3,
        #                                        .3,
        #                                        .9,
        #                                        .3,
        #                                        3,
        #                                        0.5,
        #                                        span_ratio_separately_each_epoch,
        #                                        mask_bar_num_ratio=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #                                        mask_track_num_ratio=[.5,.25,.25],
        #                                        mask_bar_ctrl_token=False,
        #                                        pretraining=False,
        #                                        train_jointly=True)


        train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                       collate_fn=lambda batch: collate_mlm(batch),num_workers=0,pin_memory=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                                       collate_fn=lambda batch: collate_mlm(batch),num_workers=0,pin_memory=True)

        # test_data_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
        #                               collate_fn=lambda batch: collate_mlm(batch))

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

        # if is_debug and checkpoint_dir:
        #
        #     optim.load_state_dict(optimizer_state)
        #
        #     # for state in optim.state.values():
        #     #     for k, v in state.items():
        #     #         if torch.is_tensor(v):
        #     #             state[k] = v.to(device)
        #
        #     weight[1] = 1
        ce_weight = torch.ones(vocab.vocab_size, device=device)
        ce_weight[0] = 0
        ce_weight[1] = config['eos_weight']
        ce_weight[7:234] = 0
        ce_loss = nn.CrossEntropyLoss(ignore_index=0, weight=ce_weight,reduction='none')

        ce_weight_all = torch.ones(vocab.vocab_size, device=device)
        ce_weight_all[0] = 0
        ce_weight_all[1] = config['eos_weight']
        ce_weight_all[7:234] = config['control_token_weight']



        if config['ce_loss_only'] is True:
            logger.info('ce only loss')
            time_signature_weight = torch.zeros(vocab.vocab_size, device=device)
            time_signature_weight[7:11] = config['control_token_weight']
            time_signature_loss = nn.CrossEntropyLoss(ignore_index=0, weight=time_signature_weight, reduction='none')

            tempo_weight = torch.zeros(vocab.vocab_size, device=device)
            tempo_weight[11:18] = config['control_token_weight']
            tempo_loss = nn.CrossEntropyLoss(ignore_index=0, weight=tempo_weight,reduction='none')

            program_weight = torch.zeros(vocab.vocab_size, device=device)
            program_weight[18:146] = config['control_token_weight']
            program_loss = nn.CrossEntropyLoss(ignore_index=0, weight=program_weight, reduction='none')

            key_weight = torch.zeros(vocab.vocab_size, device=device)
            key_weight[146:170] = config['control_token_weight']
            key_loss = nn.CrossEntropyLoss(ignore_index=0, weight=key_weight, reduction='none')

            density_weight = torch.zeros(vocab.vocab_size, device=device)
            density_weight[170:180] = config['control_token_weight']
            density_loss = nn.CrossEntropyLoss(ignore_index=0, weight=density_weight,reduction='none')

            occupation_weight = torch.zeros(vocab.vocab_size, device=device)
            occupation_weight[180:190] = 1
            occupation_loss = nn.CrossEntropyLoss(ignore_index=0, weight=occupation_weight,reduction='none')

            polyphony_weight = torch.zeros(vocab.vocab_size, device=device)
            polyphony_weight[190:200] = config['control_token_weight']
            polyphony_loss = nn.CrossEntropyLoss(ignore_index=0, weight=polyphony_weight,reduction='none')

            pitch_register_weight = torch.zeros(vocab.vocab_size, device=device)
            pitch_register_weight[200:210] = config['control_token_weight']
            pitch_register_loss = nn.CrossEntropyLoss(ignore_index=0, weight=pitch_register_weight,reduction='none')

            tensile_weight = torch.zeros(vocab.vocab_size, device=device)
            tensile_weight[210:222] = config['control_token_weight']
            tensile_loss = nn.CrossEntropyLoss(ignore_index=0, weight=tensile_weight,reduction='none')

            diameter_weight = torch.zeros(vocab.vocab_size, device=device)
            diameter_weight[222:234] = config['control_token_weight']
            diameter_loss = nn.CrossEntropyLoss(ignore_index=0, weight=diameter_weight,reduction='none')

        else:
            logger.info(f'with ordinal loss, distance is {config.distance}')
            tempo_loss = OrdinalLoss((11, 17), vocab.vocab_size,distance=config.distance,device=device)

            density_loss = OrdinalLoss((170, 179), vocab.vocab_size,distance=config.distance,device=device)
            occupation_loss = OrdinalLoss((180, 189), vocab.vocab_size,distance=config.distance,device=device)
            polyphony_loss = OrdinalLoss((190, 199), vocab.vocab_size,distance=config.distance,device=device)
            pitch_register_loss = OrdinalLoss((200, 209), vocab.vocab_size,distance=config.distance,device=device)

            tensile_loss = OrdinalLoss((210, 221), vocab.vocab_size,distance=config.distance,device=device)
            diameter_loss = OrdinalLoss((222, 233), vocab.vocab_size,distance=config.distance,device=device)

        criteria = [ce_loss, time_signature_loss, program_loss,key_loss, tempo_loss,density_loss,occupation_loss,polyphony_loss,
                      pitch_register_loss,tensile_loss,diameter_loss]

        print_every = 100

        generation_interval = 3000
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
        wandb.watch(model, criteria, log="all", log_freq=print_every)
        log_step = 0
        scheduler_optim = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=10, factor=0.5, min_lr=0.0000001, verbose=True)
        for epoch in range(start_epoch,config.epochs):


            # pbar = tqdm(total=print_every, leave=False)

            every_print_accuracy = {'total': 0,
                                    'track_control': 0,
                                    'bar_control': 0}
            for token_type in set(vocab.token_class_ranges.values()):
                every_print_accuracy[token_type] = 0

            example_ct = 0
            total_loss = 0
            ce_losses = 0
            time_signature_losses = 0
            program_losses = 0
            key_losses = 0
            tempo_losses = 0
            density_losses = 0
            occupation_losses = 0
            polyphony_losses = 0
            pitch_register_losses = 0
            tensile_losses = 0
            diameter_losses = 0

            # validate(valid_data_loader, model, config, ce_weight_all, ce_loss, time_signature_loss, program_loss,
            #          key_loss, tempo_loss, density_loss, occupation_loss, polyphony_loss, pitch_register_loss,
            #          tensile_loss, diameter_loss, device, vocab, logger)

            for step, data in enumerate(train_data_loader):
                total_step += 1
                if data is None:
                    continue
                example_ct += len(data['input'])

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
                # print('src size',src.size())
                # print('tgt_inp size',tgt_inp.size())
                outputs,weights = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
                # logger.info(src.size())
                # logger.info(tgt.size())
                # logger.info("outside model, output_size:", outputs.size())

                loss_input_1 = rearrange(outputs, 'b t v -> (b t) v')
                loss_input_2 = rearrange(tgt_out, 'b o -> (b o)')
                # loss_all = ce_loss_all(loss_input_1, loss_input_2)
                # loss_all_mean = ce_loss_all_mean(loss_input_1, loss_input_2)
                # loss_all_none = ce_loss_all_no_reduction(loss_input_1, loss_input_2)
                loss1 = ce_loss(loss_input_1, loss_input_2)
                loss1 = torch.sum(loss1) / ce_weight_all[loss_input_2].sum()
                loss2 = time_signature_loss(loss_input_1, loss_input_2)
                loss3 = program_loss(loss_input_1, loss_input_2)
                loss4 = key_loss(loss_input_1, loss_input_2)
                loss5 = tempo_loss(loss_input_1, loss_input_2)
                loss6 = density_loss(loss_input_1, loss_input_2)
                loss7 = occupation_loss(loss_input_1, loss_input_2)
                loss8 = polyphony_loss(loss_input_1, loss_input_2)
                loss9 = pitch_register_loss(loss_input_1, loss_input_2)
                loss10 = tensile_loss(loss_input_1, loss_input_2)
                loss11 = diameter_loss(loss_input_1, loss_input_2)

                if config['ce_loss_only'] is True:
                    loss2 = torch.sum(loss2) / ce_weight_all[loss_input_2].sum()
                    loss3 = torch.sum(loss3) / ce_weight_all[loss_input_2].sum()
                    loss4 = torch.sum(loss4) / ce_weight_all[loss_input_2].sum()
                    loss5 = torch.sum(loss5) / ce_weight_all[loss_input_2].sum()
                    loss6 = torch.sum(loss6) / ce_weight_all[loss_input_2].sum()
                    loss7 = torch.sum(loss7) / ce_weight_all[loss_input_2].sum()
                    loss8 = torch.sum(loss8) / ce_weight_all[loss_input_2].sum()
                    loss9 = torch.sum(loss9) / ce_weight_all[loss_input_2].sum()
                    loss10 = torch.sum(loss10) / ce_weight_all[loss_input_2].sum()
                    loss11 = torch.sum(loss11) / ce_weight_all[loss_input_2].sum()


                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + \
                       loss7 + loss8 + loss9 + loss10 + loss11


                # Backpropagate and update optim
                loss.backward()

                # optim.step_and_update_lr()
                optim.step()

                total_loss += loss.item()
                ce_losses += loss1.item()
                time_signature_losses += loss2.item()
                program_losses += loss3.item()
                key_losses += loss4.item()
                tempo_losses += loss5.item()
                density_losses += loss6.item()
                occupation_losses += loss7.item()
                polyphony_losses += loss8.item()
                pitch_register_losses += loss9.item()
                tensile_losses += loss10.item()
                diameter_losses += loss11.item()

                train_losses.append(loss.item())

                if step % print_every == print_every - 1:
                    log_step += 1
                    # if step % learning_rate_adjust_interval == learning_rate_adjust_interval - 1:
                    #     scheduler_optim.step(total_loss / print_every)
                    # pbar.close()
                    times = int((step / (print_every - 1)))

                    src_token = []

                    for i, output in enumerate(src[0]):
                        output_token = vocab.index2char(output.item())
                        src_token.append(output_token)


                    accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)

                    for token_type in accuracies.keys():
                        if accuracies[token_type] is not None:
                            every_print_accuracy[token_type] += accuracies[token_type]


                    wandb.log({"epoch": epoch,
                              "train_loss":loss,
                              "ce_loss": loss1,
                              "time_signature_loss":loss2,
                              "program_loss":loss3,
                              "key_loss":loss4,
                              "tempo_loss": loss5,
                              "density_loss": loss6,
                              "occupation_loss": loss7,
                              "polyphony_loss":loss8,
                              "pitch_register_loss":loss9,
                              "tensile_loss":loss10,
                              "diameter_loss":loss11,
                              "total_accuracy": every_print_accuracy["total"] / times,
                              "lr":optim.param_groups[0]['lr'],
                              "real_batch_num":example_ct,
                              },step=log_step)

                    logger.info(f'Epoch [{epoch+1} / {num_epochs}] \t Step [{step + 1} / {len(train_data_loader)}] \n \
                                train loss: {total_loss / print_every} \n \
                                total accuracy: {every_print_accuracy["total"] / times} \n \
                                ce loss: {ce_losses / print_every} \n \
                                time signature loss: {time_signature_losses / print_every} \n \
                                 program loss : {program_losses / print_every} \n \
                                key loss: {key_losses / print_every}\n \
                                tempo loss: {tempo_losses / print_every} \n \
                                density loss : {density_losses / print_every} \n \
                                occupation loss: {occupation_losses / print_every}\n \
                                polyphony loss : {polyphony_losses / print_every}\n \
                                pitch_register loss {pitch_register_losses / print_every}\n \
                                tensile loss {tensile_losses / print_every}\n \
                                diameter loss {diameter_losses / print_every}')

                    if lr != optim.param_groups[0]['lr']:
                        lr = optim.param_groups[0]['lr']
                        logger.info(f'learning rate is {lr}')

                    for token_type in every_print_accuracy.keys():
                        logger.debug(f'{token_type} accuracy is {every_print_accuracy[token_type] / times}')

                    for token_type in every_print_accuracy.keys():
                        wandb.log({f'{token_type}_acc': every_print_accuracy[token_type] / times,
                                  'real_batch_num': example_ct,
                                   },step=log_step)

                    total_loss = 0
                    ce_losses = 0
                    time_signature_losses = 0
                    program_losses = 0
                    key_losses = 0
                    tempo_losses = 0
                    density_losses = 0
                    occupation_losses = 0
                    polyphony_losses = 0
                    pitch_register_losses = 0
                    tensile_losses = 0
                    diameter_losses = 0


                    # logger.info(f'Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_data_loader)}] \n '
                    #             f'Train Loss: {total_loss / print_every} \t Accuracy {accuracies["total"] / times} \n'
                if step % (print_every*10) == print_every*10 - 1:
                    # wandb.log({'input_size':src.size(),
                    #            'input':src_token[:50],
                    #             'output_size': len(target_output),
                    #              'generated_output': generated_output[:50],
                    #              'target_output': target_output[:50],
                    #              })

                    logger.debug(f'input size is {src.size()} \n'
                                 f'input is : {src_token[:50]} \n'
                                 f'output size is {len(target_output)} \n'
                                 f'generated output: {generated_output[:50]} \n'
                                 f'target output: {target_output[:50]} \n'
                                 )

                    # pbar = tqdm(total=print_every, leave=False)
                if step % generation_interval == generation_interval - 1:
                    batch_number = random.choice(range(len(test_batches)))
                    in_batch_number = random.choice(range(len(test_batches[batch_number])))
                    logger.debug(f'test batch {(batch_number,in_batch_number)}')
                    event = test_batches[batch_number][in_batch_number]

                    # r = re.compile('i_\d')
                    #
                    # track_program = list(filter(r.match, event))
                    # track_nums = len(track_program)
                    #
                    # if track_nums == 1:

                    generate_and_prediction(model,event,device,vocab)

            logger.info(f'Epoch [{epoch} / {num_epochs} end]')

            for token_type in train_accuracies.keys():
                train_accuracies[token_type] = (every_print_accuracy[token_type] / times)
                wandb.log({
                    f'ave_epoch_train_{token_type}_acc': train_accuracies[token_type],
                    'epoch_metrics_step': epoch},step=log_step
                    )
                logger.info(f'ave_epoch_train_{token_type}_acc is {train_accuracies[token_type]}')

            average_train_loss = np.mean(train_losses)
            scheduler_optim.step(average_train_loss)

            wandb.log({
                        'ave_epoch_train_loss': average_train_loss,
                        'epoch_metrics_step':epoch},step=log_step
                        )

            logger.info(
                        f'average train losses is {average_train_loss} \t'
                        )

            # Validate every epoch
            # pbar.close()
            logger.info(f'valid data size is {len(valid_data_loader)}')
            val_loss, val_accuracy = validate(valid_data_loader, model, config, ce_weight_all, ce_loss,time_signature_loss,program_loss,key_loss,tempo_loss,density_loss,occupation_loss,polyphony_loss,pitch_register_loss,tensile_loss,diameter_loss, device, vocab, logger)

            for key in val_loss.keys():
                if key in ['total','ce_loss','density','occupation','polyphony',
                           'pitch_register','tensile','diameter','tempo','time_signature',
                           'key','program']:
                    wandb.log({f'val_{key}_loss': val_loss[key],
                               'epoch_metrics_step': epoch},step=log_step)
                    logger.info(f'validation {key} loss is {val_loss[key]}')

            for key in val_accuracy.keys():

                wandb.log({f'val_{key}_accuracy': val_accuracy[key],
                           'epoch_metrics_step': epoch},step=log_step)
                logger.info(f'validation {key} accuracy is {val_accuracy[key]}')

            # val_losses.append(val_loss)


            path = os.path.join(wandb.run.dir, f"checkpoint_{epoch}")
            logger.info(f'checkpoint_dir is {path}')

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'epoch': epoch,
                        'loss': val_loss['total']}, path)

            # tune.report(train_loss=average_train_loss,
            #             train_accuracy=train_accuracies['total'],
            #             lr=lr,
            #             loss=val_loss,
            #             val_loss=val_loss,
            #             val_accuracy=val_accuracy['total'],
            #             pitch_accuracy=val_accuracy['pitch'],
            #             duration_accuracy=val_accuracy['duration'],
            #             structure_accuracy=val_accuracy['structure'],
            #             tempo_accuracy=val_accuracy['tempo'],
            #             time_signature_accuracy=val_accuracy['time_signature'],
            #             program_accuracy=val_accuracy['program'],
            #             eos_accuracy=val_accuracy['eos'],
            #             track_control_accuracy=val_accuracy['track_control'],
            #             bar_control_accuracy=val_accuracy['bar_control'],
            #             density_accuracy=val_accuracy['density'],
            #             occupation_accuracy=val_accuracy['occupation'],
            #             polyphony_accuracy=val_accuracy['polyphony'],
            #             pitch_register_accuracy=val_accuracy['pitch_register'],
            #             tensile_accuracy=val_accuracy['tensile'],
            #             diameter_accuracy=val_accuracy['diameter'],
            #             key_accuracy=val_accuracy['key'],
            #             )

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
def generation_accuracy(outputs, targets, vocab):
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

        for i, output in enumerate(outputs):


            target_classes = vocab.get_token_classes(vocab.char2index(targets[i]))

            accuracy_result[target_classes] += outputs[i] == targets[i]
            types_number_counter[target_classes] += 1

            accuracy_result['total'] += outputs[i] == targets[i]
            types_number_counter['total'] += 1

        accuracy_result['track_control'] = accuracy_result['density'] + \
                                           accuracy_result['occupation'] + \
                                           accuracy_result['polyphony'] + \
                                           accuracy_result['pitch_register']

        types_number_counter['track_control'] = types_number_counter['density'] + \
                                                types_number_counter['occupation'] + \
                                                types_number_counter['polyphony'] + \
                                                types_number_counter['pitch_register']


        accuracy_result['bar_control'] = accuracy_result['tensile'] + \
                                         accuracy_result['diameter']

        types_number_counter['bar_control'] = types_number_counter['tensile'] + \
                                              types_number_counter['diameter']

        for token_type in accuracy_result.keys():
            if types_number_counter[token_type] != 0:
                accuracy_result[token_type] /= types_number_counter[token_type]
            else:
                accuracy_result[token_type] = None

        accuracy_result = {k: v for k, v in accuracy_result.items() if v is not None}


        return accuracy_result





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
                # log the first one to print
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


        accuracy_result['bar_control'] = accuracy_result['tensile'] + \
                                         accuracy_result['diameter']

        types_number_counter['bar_control'] = types_number_counter['tensile'] + \
                                              types_number_counter['diameter']

        for token_type in accuracy_result.keys():
            if types_number_counter[token_type] != 0:
                accuracy_result[token_type] /= types_number_counter[token_type]
            else:
                accuracy_result[token_type] = None

        accuracy_result = {k: v for k, v in accuracy_result.items() if v is not None}

        return accuracy_result, generated_output, target_output



def validate(valid_loader, model,config, ce_weight_all,ce_loss,time_signature_loss, program_loss, key_loss,tempo_loss,density_loss,occupation_loss,polyphony_loss,pitch_register_loss,tensile_loss,diameter_loss, device, vocab, logger):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'



    total_loss = {'total': 0, 'ce_loss': 0}

    total_steps = 0

    total_data_length = len(valid_loader)

    total_accuracy = {'total': 0,
                      'track_control': 0,
                      'bar_control': 0}
    types_number_counter = {}
    types_number_counter['bar_control'] = 0
    for token_type in set(vocab.token_class_ranges.values()):
        total_accuracy[token_type] = 0
        total_loss[token_type] = 0
        types_number_counter[token_type] = 0
        types_number_counter['total'] = 0
        types_number_counter['track_control'] = 0

    for data in iter(valid_loader):
        total_steps += 1
        if data is None:
            continue
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
            outputs,weights = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)

            loss_input_1 = rearrange(outputs, 'b t v -> (b t) v')
            loss_input_2 = rearrange(tgt_out, 'b o -> (b o)')

            loss1 = ce_loss(loss_input_1, loss_input_2)
            loss1 = torch.sum(loss1) / ce_weight_all[loss_input_2].sum()
            loss2 = time_signature_loss(loss_input_1, loss_input_2)
            loss3 = program_loss(loss_input_1, loss_input_2)
            loss4 = key_loss(loss_input_1, loss_input_2)
            loss5 = tempo_loss(loss_input_1, loss_input_2)
            loss6 = density_loss(loss_input_1, loss_input_2)
            loss7 = occupation_loss(loss_input_1, loss_input_2)
            loss8 = polyphony_loss(loss_input_1, loss_input_2)
            loss9 = pitch_register_loss(loss_input_1, loss_input_2)
            loss10 = tensile_loss(loss_input_1, loss_input_2)
            loss11 = diameter_loss(loss_input_1, loss_input_2)

            if config['ce_loss_only'] is True:
                loss2 = torch.sum(loss2) / ce_weight_all[loss_input_2].sum()
                loss3 = torch.sum(loss3) / ce_weight_all[loss_input_2].sum()
                loss4 = torch.sum(loss4) / ce_weight_all[loss_input_2].sum()
                loss5 = torch.sum(loss5) / ce_weight_all[loss_input_2].sum()
                loss6 = torch.sum(loss6) / ce_weight_all[loss_input_2].sum()
                loss7 = torch.sum(loss7) / ce_weight_all[loss_input_2].sum()
                loss8 = torch.sum(loss8) / ce_weight_all[loss_input_2].sum()
                loss9 = torch.sum(loss9) / ce_weight_all[loss_input_2].sum()
                loss10 = torch.sum(loss10) / ce_weight_all[loss_input_2].sum()
                loss11 = torch.sum(loss11) / ce_weight_all[loss_input_2].sum()

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + \
                       loss7 + loss8 + loss9 + loss10 + loss11



            total_loss['total'] += loss.item()
            total_loss['ce_loss'] += loss1.item()
            total_loss['time_signature'] += loss2.item()
            total_loss['program'] += loss3.item()
            total_loss['key'] += loss4.item()
            total_loss['tempo'] += loss5.item()
            total_loss['density'] += loss6.item()
            total_loss['occupation'] += loss7.item()
            total_loss['polyphony'] += loss8.item()
            total_loss['pitch_register'] += loss9.item()
            total_loss['tensile'] += loss10.item()
            total_loss['diameter'] += loss11.item()

            accuracies, generated_output, target_output = accuracy(outputs, tgt_out, vocab)



            for token_type in total_accuracy.keys():
                if token_type in accuracies:
                    total_accuracy[token_type] += accuracies[token_type]
                    types_number_counter[token_type] += 1

            # src_token = []
            # tgt_inputs = []
            # for i, output in enumerate(src[0]):
            #     output_token = vocab.index2char(output.item())
            #     src_token.append(output_token)
            #
            # for i, output in enumerate(tgt_inp[0]):
            #     tgt_inputs.append(vocab.index2char(output.item()))
            #
            # break


    for key in total_loss.keys():

        total_loss[key] /= total_steps

    for key in total_accuracy.keys():
        if types_number_counter[key] != 0:
            total_accuracy[key] /= types_number_counter[key]
        else:
            total_accuracy[key] = None

    src_token = []
    tgt_inputs = []
    for i, output in enumerate(src[0]):
        output_token = vocab.index2char(output.item())
        src_token.append(output_token)

    for i, output in enumerate(tgt_inp[0]):
        tgt_inputs.append(vocab.index2char(output.item()))

    logger.info(f'input size is {src.size()} \n'
                f'input is : {src_token[:50]} \n'
                f'target input: {tgt_inputs} \n'
                f'output size is {len(target_output)} \n'
                f'generated output: {generated_output} \n'
                f'target output: {target_output} \n')

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
            outputs, weights = model(src, tgt_inp, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask)
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


            #test
            src_token = []
            for i, output in enumerate(src[0]):
                output_token = vocab.index2char(output.item())
                src_token.append(output_token)

            logger.info(f'input size is {src.size()} \n'
                        f'input is : {src_token[:50]} \n'
                        f'output size is {len(target_output)} \n'
                        f'generated output: {generated_output[:50]} \n'
                        f'target output: {target_output[:50]} \n')

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

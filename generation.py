import torch
import torch.nn as nn
from model import ScoreTransformer
import yaml
import argparse
from einops import rearrange
import re
import preprocessing
from preprocessing import event_2midi
import math
import os
from vocab import *
import pretty_midi
import logging
import coloredlogs
from datetime import datetime
global logger

logger = logging.getLogger(__name__)

logger.handlers = []

logfile = f'generate.log'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S', filename=logfile)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logger.addHandler(console)

coloredlogs.install(level='INFO', logger=logger, isatty=True)

span_ratio_separately_each_epoch = np.array([[1, 0, 0], [.5, .5, 0],
                                             [.25, .75, 0], [.25, .5, .25],
                                             [.25, .25, .5]])
vocab = WordVocab(all_tokens)

import tension_calculation
import dataset

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

def cal_track_control(file_events,pm):


    file_events = preprocessing.remove_control_event(file_events.tolist(),control_tokens)
    r = re.compile('i_\d')

    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)



    file_events = np.array(file_events)

    #     logger.info(f'number of bars is {len(bar_pos)}')
    #     logger.info(f'time signature is {file_event[1]}')
    bar_length = int(file_events[1][0])
    bar_pos = np.where(file_events == 'bar')[0]
    if bar_length != 6:
        bar_length = bar_length * 4 * len(bar_pos)
    else:
        bar_length = bar_length / 2 * 4 * len(bar_pos)
    #     logger.info(f'bar length is {bar_length}')

    track_events = {}

    for i in range(num_of_tracks):
        track_events[f'track_{i}'] = []
    track_names = list(track_events.keys())
    for bar_index in range(len(bar_pos) - 1):
        bar = bar_pos[bar_index]
        next_bar = bar_pos[bar_index + 1]
        bar_events = file_events[bar:next_bar]
        #         logger.info(bar_events)

        track_pos = []

        for track_name in track_names:
            if len(np.where(track_name == bar_events)[0]) == 0:
                logger.info(bar_events)
            track_pos.append(np.where(track_name == bar_events)[0][0])
        #         logger.info(track_pos)
        for track_index in range(len(track_names) - 1):
            track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
            track_events[track_names[track_index]].append(track_event)
        #             logger.info(track_event)
        else:
            track_index += 1
            track_event = bar_events[track_pos[track_index]:]
            #             logger.info(track_event)
            track_events[track_names[track_index]].append(track_event)

    densities = dataset.note_density(track_events, bar_length)
    density_category = dataset.to_category(densities, dataset.control_bins)

    occupation_rate, polyphony_rate = dataset.occupation_polyphony_rate(pm)
    occupation_category = dataset.to_category(occupation_rate, dataset.control_bins)
    polyphony_category = dataset.to_category(polyphony_rate, dataset.control_bins)
    pitch_register_category = dataset.pitch_register(track_events)
    #     logger.info(densities)
    #     logger.info(occupation_rate)
    #     logger.info(polyphony_rate)
    #     logger.info(density_category)
    #     logger.info(occupation_category)
    #     logger.info(polyphony_category)

    #     key_token =  key_to_token[key]

    density_token = [f'd_{category}' for category in density_category]
    occupation_token = [f'o_{category}' for category in occupation_category]
    polyphony_token = [f'y_{category}' for category in polyphony_category]
    pitch_register_token = [f'r_{category}' for category in pitch_register_category]

    track_control_tokens = density_token + occupation_token + polyphony_token + pitch_register_token

    logger.info(track_control_tokens)
    return track_control_tokens







def cal_bar_control(target_name, generated_name):


    result = tension_calculation.extract_notes(pretty_midi.PrettyMIDI(target_name), 3)


    pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result

    key_name = tension_calculation.all_key_names

    result_target = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices,  -1, key_name)

    total_tension_target, diameters_target, key_name_target = result_target

    target_tensile_category = dataset.to_category(total_tension_target, dataset.tensile_bins)
    target_diameter_category = dataset.to_category(diameters_target, dataset.tensile_bins)

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



    logger.info(f'target key {key_name_target}, generated key {key_name_generated}')

    return generated_tensile_category, generated_diameter_category,\
           target_tensile_category, target_diameter_category,\
           key_name_generated, key_name_target


def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


def get_mask_track_name(mask_tracks,mask_idx):
    total_tracks = 0
    for one_bar_tracks in mask_tracks:
        for track in one_bar_tracks:
            if total_tracks == mask_idx:
                if track == 0:
                    #tensile
                    return 's_'
                if track == 1:
                    #diameter
                    return 'a_'
                return f'track_{track-2}'
            else:
                total_tracks += 1

def get_bar_duration(meter):
    if meter == 6:
        return 3
    else:
        return meter


def total_duration(duration_list,duration_name_to_time):
    total = 0
    if duration_list:

        for duration in duration_list:
            total += duration_name_to_time[duration]
    return total

def clear_pitch_duration_event(
                               curr_time,
                               previous_duration,
                               is_rest_s,
                               duration_list,
                               duration_name_to_time):
    if is_rest_s:
        duration = total_duration(duration_list,duration_name_to_time)
        curr_time -= previous_duration

    else:
        duration = total_duration(duration_list,duration_name_to_time)

    curr_time += duration
    previous_duration = duration

    return curr_time,previous_duration

def prediction(model,event,token_type,device):
    src, tgt_out = mask_category(event,token_type,mask_num=3,pos=0)
    src_token = []


    src.to(device)
    tgt_out.to(device)

    target_output = []
    for i, token_idx in enumerate(tgt_out):
        target_token = vocab.index2char(token_idx.item())
        target_output.append(target_token)
    logger.info('target output is:')
    for token in target_output:
        if token[0] == 'k':
            logger.info(f'target key is {all_key_names[int(token[2:])]}')
        else:
            logger.info(token)

    for i, token_idx in enumerate(src):
        src_token.append(vocab.index2char(token_idx.item()))

    src_masked_times = torch.sum(src == vocab.char2index('m_0')).item()
    tgt_inp = []
    tgt_inp_token = []
    weight_list = []
    generate_times = 0

    with torch.no_grad():
        mask_idx = 0

        while mask_idx < src_masked_times:
            tgt_inp.append(vocab.char2index('m_0'))

            sampling_times = 0
            output,weights = model_generate(model, src, tgt_inp,device=device,return_weights=True)
            index = sampling(output[-1], 0.9)
            sampling_times += 1

            output_token = vocab.index2char(index)
            # logger.info(output_token)
                # logger.info(event)

            while vocab.token_class_ranges[index] not in token_type and sampling_times < 10:
                index = sampling(output[-1], 0.9)
                output_token = vocab.index2char(index)
                sampling_times += 1



            if sampling_times < 10:
                tgt_inp.append(index)
                mask_idx += 1
                weight_list.append(weights)
            else:
                logger.info(f'track name is not equal, generate again, generate time is {generate_times}')
                generate_times += 1
                tgt_inp.pop()
                if generate_times > 10:
                    logger.info(f'fail to have correct track name {mask_idx} after 10 times')
                    tgt_inp.append(index)
                    mask_idx += 1

                # i += 1
    generated_output = []



    for i, token_idx in enumerate(tgt_inp):
        tgt_inp_token.append(vocab.index2char(token_idx))
        if token_idx != 2:
            output_token = vocab.index2char(token_idx)
            generated_output.append(output_token)



    logger.info('generated output is:')
    for token in generated_output:
        if token[0] == 'k':
            logger.info(f'generated key is {all_key_names[int(token[2:])]}')
        else:
            logger.info(token)



    # logger.info(f'generation {generated_output}')
    # logger.info(f'target {target_output}')


    return src_token, generated_output, target_output, tgt_inp_token,weight_list


def generate(model,event,mask_bars,mask_tracks,mask_tensile,mask_diameter,device):
    src, tgt_out,mask_bars,mask_tracks = mask_bar_and_track(event, mask_bars=mask_bars, mask_tracks=mask_tracks,mask_tensile=mask_tensile,mask_diameter=mask_diameter)


    src.to(device)
    tgt_out.to(device)

    duration_name_to_time, duration_time_to_name, duration_times, bar_duration = preprocessing.get_note_duration_dict(
        1, (int(event[1][0]), int(event[1][2])))

    src_masked_track = torch.sum(src == vocab.char2index('m_0')).item()
    tgt_inp = []
    generate_times = 0


    with torch.no_grad():
        mask_idx = 0
        while mask_idx < src_masked_track:
            logger.info(f'current mask idx is {mask_idx}')

        # for mask_idx in range(src_masked_track):
            this_tgt_inp = []

            this_tgt_failure = False
            mask_track_name = get_mask_track_name(mask_tracks,mask_idx)
            # i = 0
            this_tgt_inp.append(vocab.char2index('m_0'))

            curr_time = 0
            previous_duration = 0

            in_duration_event = False
            is_rest_s = False

            in_pitch_event = False

            duration_list = []

            while this_tgt_inp[-1] != vocab.char2index('<eos>') and len(this_tgt_inp) < 500:
                sampling_times = 0
                output = model_generate(model, src, tgt_inp + this_tgt_inp,device)
                index = sampling(output[-1], 0.9)
                sampling_times += 1

                event = vocab.index2char(index)
                # logger.info(event)

                if len(this_tgt_inp) == 1:
                    if mask_track_name in ['s_','a_']:
                        compare_event = event[:2]
                    else:
                        compare_event = event
                    if compare_event != mask_track_name:
                        while compare_event != mask_track_name and sampling_times < 10:
                            index = sampling(output[-1], 0.9)
                            event = vocab.index2char(index)
                            sampling_times += 1
                            if mask_track_name in ['s_', 'a_']:
                                compare_event = event[:2]
                            else:
                                compare_event = event
                        if compare_event != mask_track_name:
                            sampling_times = 0
                            while compare_event != mask_track_name and sampling_times < 10:
                                output = model_generate(model, src, tgt_inp + this_tgt_inp,device)
                                index = sampling(output[-1], 0.9)
                                event = vocab.index2char(index)
                                sampling_times += 1
                                if mask_track_name in ['s_', 'a_']:
                                    compare_event = event[:2]
                                else:
                                    compare_event = event
                            if compare_event != mask_track_name:
                                logger.info(f'{compare_event} is not equal to {mask_track_name}')
                                logger.info(f'mask {mask_idx} needs to be generated again')
                                this_tgt_failure = True


                if event in pitches:
                    in_pitch_event = True
                    if in_duration_event:
                        curr_time, previous_duration = clear_pitch_duration_event(
                            curr_time,
                            previous_duration,
                            is_rest_s,
                            duration_list,duration_name_to_time)

                        duration_list = []

                        in_duration_event = False
                        is_rest_s = False

                if event in duration_name_to_time.keys():
                    if not in_duration_event and not in_pitch_event:
                        # generate a pitch event token
                        while event not in pitches and sampling_times < 10:
                            index = sampling(output[-1], 0.9)
                            event = vocab.index2char(index)
                            sampling_times += 1
                        if event not in pitches:
                            sampling_times = 0
                            while event not in pitches and sampling_times < 10:
                                output = model_generate(model, src, tgt_inp + this_tgt_inp,device)
                                index = sampling(output[-1], 0.9)
                                event = vocab.index2char(index)
                                sampling_times += 1
                            if event != pitches:
                                logger.info(f'{event} is not pitch')
                                logger.info(f'mask {mask_idx} needs to be generated again')
                                this_tgt_failure = True

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
                            duration_list,duration_name_to_time)

                    if mask_track_name not in ['s_','a_']:
                        if not math.isclose(curr_time,bar_duration):
                            logger.info(f'{curr_time} is not equal to {bar_duration}')
                            logger.info(f'mask {mask_idx} needs to be generated again')
                            this_tgt_failure = True

                this_tgt_inp.append(index)
            if this_tgt_inp[-1] == vocab.char2index('<eos>') and not this_tgt_failure:
                mask_idx += 1
                tgt_inp.extend(this_tgt_inp[:-1])
            else:
                logger.info(f'generate again, generate time is {generate_times}')
                generate_times += 1
                if generate_times > 10:
                    logger.info(f'fail to have correct track duration {mask_idx} after 10 times')
                    return None
                    tgt_inp.extend(this_tgt_inp)
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
    return restore_marked_input(src_token,generated_output,target_output), mask_bars



def cal_duration(events,time_signature):
    duration_name_to_time, duration_time_to_name, duration_times, bar_duration = preprocessing.get_note_duration_dict(
        1, (time_signature[0], time_signature[2]))

    def total_duration(duration_list):
        total = 0
        if duration_list:

            for duration in duration_list:
                total += duration_name_to_time[duration]
        return total

    def clear_pitch_duration_event(
                                   curr_time,
                                   previous_duration,
                                   is_rest_s,
                                   duration_list):
        if is_rest_s:
            duration = total_duration(duration_list)
            curr_time -= previous_duration

        else:
            duration = total_duration(duration_list)

        curr_time += duration
        previous_duration = duration

        return curr_time,previous_duration


    for i, event in enumerate(events):


        if event in control_tokens:
            continue

        if event in duration_name_to_time.keys():
            duration_list.append(event)
            in_duration_event = True
            continue

        # an event not in duration event happens
        if in_duration_event:
            curr_time, previous_duration = clear_pitch_duration_event(
                                                                      curr_time,
                                                                      previous_duration,
                                                                      is_rest_s,
                                                                      duration_list)

            duration_list = []

            in_duration_event = False
            is_rest_s = False


        if event == 'rest_s':
            is_rest_s = True


        if event == '<eos>':
            return curr_time







def bar_track_validate(output_indices, index, mask_track_name, time_signature,total_duration):
    if len(output_indices) == 1:
        # index must match track name
        if vocab.char2index(index) != mask_track_name:
            return False
    if vocab.char2index(index) == '<eos>':
        duration = cal_duration(output_indices,time_signature)
        if total_duration != duration:
            return False
    return True




def restore_marked_input(src_token, generated_output, target_output):
    src_token = np.array(src_token,dtype='<U9')

    # restore with generated output
    restored_with_generated_token = src_token.copy()

    generated_output = np.array(generated_output)
    target_output = np.array(target_output)
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

    # restore with target
    r = re.compile('(?:track_\d|a_|s_)')

    target_mask_names = list(filter(r.match, target_output))
    target_output = np.array(target_output)
    if '<pad>' in target_output:
        end_index = np.where(target_output == '<pad>')[0][0]
    else:
        end_index = len(target_output)
    target_output = target_output.tolist()
    start_index = 0
    target_mask_indices = []
    if len(set(target_mask_names)) == 1:
        for name in target_mask_names:
            result_pos = target_output.index(name, start_index, end_index)
            target_mask_indices.append(result_pos)
            start_index = result_pos+1

    else:
        for name in target_mask_names:
            result_pos = target_output.index(name, start_index, end_index)
            target_mask_indices.append(result_pos)
            start_index = result_pos
    # logger.info(target_mask_indices)
    restored_with_target_token = src_token.copy()

    if len(target_mask_indices) == 1:
        mask_indices = np.where(restored_with_target_token == 'm_0')[0]
        target_result_sec = target_output[target_mask_indices[0]:end_index - 1]

        #         logger.info(len(target_result_sec))
        restored_with_target_token = np.delete(restored_with_target_token, mask_indices[0])
        for token in target_result_sec[::-1]:
            #             logger.info(token)
            restored_with_target_token = np.insert(restored_with_target_token, mask_indices[0], token)

    else:
        for i in range(len(target_mask_indices) - 1):

            mask_indices = np.where(restored_with_target_token == 'm_0')[0]
            target_result_sec = target_output[target_mask_indices[i]:target_mask_indices[i + 1] - 1]

            #             logger.info(len(target_result_sec))
            #             logger.info(mask_indices[i])
            restored_with_target_token = np.delete(restored_with_target_token, mask_indices[0])
            for token in target_result_sec[::-1]:
                #                 logger.info(token)
                restored_with_target_token = np.insert(restored_with_target_token, mask_indices[0], token)

        else:
            mask_indices = np.where(restored_with_target_token == 'm_0')[0]
            target_result_sec = target_output[target_mask_indices[i + 1]:end_index - 1]

            #             logger.info(len(target_result_sec))
            restored_with_target_token = np.delete(restored_with_target_token, mask_indices[0])
            for token in target_result_sec[::-1]:
                #                 logger.info(token)
                restored_with_target_token = np.insert(restored_with_target_token, mask_indices[0], token)

                # restore with rest

    if src_token[1] == '4/4':
        fill_in_tokens = ['rest_e', 'whole']
    elif src_token[1] == '2/4':
        fill_in_tokens = ['rest_e', 'half']
    else:
        fill_in_tokens = ['rest_e', 'half', 'quarter']

    mask_indices = np.where(src_token == 'm_0')[0]
    restored_empty_fill = src_token.copy()

    for i in range(len(mask_indices)):

        # logger.info(target_mask_names)
        mask_indices = np.where(restored_empty_fill == 'm_0')[0]


        # logger.info(mask_indices)
        restored_empty_fill = np.delete(restored_empty_fill, mask_indices[0])
        if target_mask_names[i][0] == 't':
            for token in fill_in_tokens[::-1]:
                # logger.info(token)
                restored_empty_fill = np.insert(restored_empty_fill, mask_indices[0], token)
                # logger.info(target_mask_names[i])
        restored_empty_fill = np.insert(restored_empty_fill, mask_indices[0], target_mask_names[i])



    # for i, index in enumerate(mask_indices):
    #     if index > bar_pos[-1]:
    #         logger.info(f'masked bar {len(bar_pos)} {track_names[i]}')
    #     else:
    #         bar_start_index = np.where(index < bar_pos)[0][0]
    #         logger.info(f'masked bar {bar_start_index} {track_names[i]}')

    return restored_with_generated_token, restored_with_target_token, restored_empty_fill


def change_control(batch, control_change, control_tokens,mininum,maximum,mask_bars=None,mask_tracks=None):
    new_tokens = []

    if control_tokens[0] in bar_control_tokens:
        if mask_bars:
            cur_bar = 0

            for token in batch:
                if token in control_tokens:
                    if cur_bar in mask_bars:
                        token_new_category = int(token[2:]) + control_change
                        if token_new_category < mininum:
                            token_new_category = 0
                        if token_new_category > maximum:
                            token_new_category = maximum
                        token_new_category = str(token_new_category)
                        token = token[:2] + token_new_category
                    cur_bar += 1

                new_tokens.append(token)
        else:
            for token in batch:
                if token in control_tokens:

                    token_new_category = int(token[2:]) + control_change
                    if token_new_category < mininum:
                        token_new_category = 0
                    if token_new_category > maximum:
                        token_new_category = maximum
                    token_new_category = str(token_new_category)
                    token = token[:2] + token_new_category
                new_tokens.append(token)


    if control_tokens[0] in track_control_tokens:
        r = re.compile('i_\d')

        track_program = list(filter(r.match, batch))
        track_nums = len(track_program)

        track_num = 0
        for token in batch:
            if token in control_tokens:
                if mask_tracks:
                    if track_num in mask_tracks:
                        token_new_category = int(token[2:]) + control_change
                        if token_new_category < mininum:
                            token_new_category = 0
                        if token_new_category > maximum:
                            token_new_category = maximum
                        token_new_category = str(token_new_category)
                        token = token[:2] + token_new_category
                    track_num += 1
                    if track_num == track_nums:
                        track_num = 0
                else:
                    token_new_category = int(token[2:]) + control_change
                    if token_new_category < mininum:
                        token_new_category = 0
                    if token_new_category > maximum:
                        token_new_category = maximum
                    token_new_category = str(token_new_category)
                    token = token[:2] + token_new_category

            new_tokens.append(token)



    return new_tokens


# mask song/track/bar control token, span=1 for all
def mask_category(event, token_type,mask_num=1,pos=0):
    tokens = []
    decoder_in = []
    decoder_target = []
    start_pos = 0
    total_masked_ratio = 0
    masked_num = 0
    mask_pos = 0
    while start_pos < len(event):
        if vocab.token_class_ranges[vocab.char2index(event[start_pos])] in token_type  and masked_num < mask_num:
            if mask_pos >= pos:
                masked_token = event[start_pos]
                tokens.append(vocab.mask_indices[0])
                decoder_target.append(vocab.char2index(masked_token))
                masked_num += 1
            else:
                tokens.append(vocab.char2index(event[start_pos]))

            mask_pos += 1
            total_masked_ratio += 1 / len(event)
            start_pos += 1

        else:
            tokens.append(vocab.char2index(event[start_pos]))
            start_pos += 1
    tokens = np.array(tokens)

    if len(decoder_target) > 0:
        decoder_target = np.array(decoder_target)
    else:
        logger.info('no masked token')
        return None

    return torch.tensor(tokens).long(),torch.tensor(decoder_target).long()



def mask_bar_and_track(event, mask_bars=None, mask_tracks=None,mask_tensile=None,mask_diameter=None):
    # mask bar token (w/wo bar control token) and try to generate bar token

    total_tokens = []
    total_decoder_in = []
    total_decoder_target = []

    tokens = []
    decoder_in = []
    decoder_target = []
    masked_indices_pairs = []

    bar_pos = np.where(np.array(event) == 'bar')[0]



    r = re.compile('i_\d')

    track_program = list(filter(r.match, event))
    track_nums = len(track_program)

    if track_nums == 3:
        #         ratios = track_ratio[0]
        track_0_pos = np.where('track_0' == np.array(event))[0]
        track_1_pos = np.where('track_1' == np.array(event))[0]
        track_2_pos = np.where('track_2' == np.array(event))[0]
        all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos, track_2_pos]))

    else:
        #         ratios = track_ratio[1]
        track_0_pos = np.where('track_0' == np.array(event))[0]
        track_1_pos = np.where('track_1' == np.array(event))[0]
        all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos]))

    #     bar_number_prob = np.array(bar_ratio[:len(bar_pos)],dtype=float)
    #     if not np.any(bar_number_prob):
    #         bar_number_prob = np.ones(len(bar_pos))

    #     logger.info(f'bar number prob is {bar_number_prob}')
    #     bar_number_prob /= np.sum(bar_number_prob)

    #     # if np.sum(bar_number_prob) != 1:
    #     #     logger.info(bar_number_prob)
    #     # logger.info(f'bar number prob is {bar_number_prob}')
    #     resample_counts = np.random.multinomial(1, bar_number_prob[:len(bar_pos)])
    #     bar_mask_number = np.where(resample_counts)[0][0] + 1
    #     # bar_mask_number = np.random.choice(range(len(bar_pos)), size=1, p=bar_ratio[:len(bar_pos)])
    #     bar_start_indices = np.sort(np.random.choice(len(bar_pos),size=bar_mask_number,replace=False))

    if mask_bars is None:
        mask_bars = range(len(bar_pos))
        logger.info(mask_bars)
    else:
        if not isinstance(mask_bars,list):
            if mask_bars >= len(bar_pos):
                logger.info(f'mask bar {mask_bars} is larger than total bar {len(bar_pos)}, mask bar 1 instead')
                mask_bars  = 1
            mask_bars = [mask_bars]
        else:
            if len(mask_bars) >= len(bar_pos):
                mask_bars = mask_bars[0:len(bar_pos)]

    for i, bar in enumerate(mask_bars):
        if bar > len(bar_pos) - 1:
            logger.info(f'mask bar {bar} is larger than total bar {len(bar_pos)}, mask bar {i} instead')
            mask_bars[i] = i

    mask_bars = list(dict.fromkeys(mask_bars))

    if mask_tracks is None:
        mask_track = [i+2 for i in range(track_nums)]
        mask_tracks = [mask_track for _ in mask_bars]
    else:
        if not isinstance(mask_tracks,list):
            mask_tracks = [mask_tracks+2]
        else:
            mask_tracks = [track + 2 for track in mask_tracks]
        if len(mask_tracks) != len(mask_bars):
            mask_tracks = [mask_tracks for _ in mask_bars]

    if mask_tensile is not None:
        if len(mask_tensile) != len(mask_bars):
            mask_tensile = [mask_tensile[0] for _ in mask_bars]


    if mask_diameter is not None:
        if len(mask_diameter) != len(mask_bars):
            mask_diameter = [mask_diameter[0] for _ in mask_bars]



    # logger.info(f'bar start indices is {bar_start_indices}')
    for i, bar_num in enumerate(mask_bars):
        # bar_start_index = np.random.choice(len(bar_pos))

        if bar_num == len(bar_pos) - 1:
            next_bar_start_pos = len(event)
        else:
            next_bar_num = bar_num + 1
            next_bar_start_pos = bar_pos[next_bar_num]

        bar_start_pos = bar_pos[bar_num]

        track_start_index = np.where(all_track_pos > bar_start_pos)[0][0]
        track_positions = all_track_pos[track_start_index:track_start_index + track_nums]

        track_positions = np.append(track_positions, next_bar_start_pos)


        start_pos = track_positions[0]
        track_positions = np.insert(track_positions, 0, start_pos - 1)
        track_positions = np.insert(track_positions, 0, start_pos - 2)
        track_with_bar_ctrl_pos = track_positions
        #         prob = random.random()
        #         # logger.info(prob)
        #         if prob < ratios[0]:
        #             # select one track
        #             track_pos_select_index = [np.random.choice(track_num)]
        #         elif prob < ratios[0] + ratios[1]:
        #             # select two tracks
        #             track_pos_select_index = np.sort(np.random.choice(track_num, 2, replace=False))
        #         else:
        #             # select three tracks
        #             track_pos_select_index = np.arange(track_num)
        # logger.info(track_nums)
        # logger.info(bar_start_index)


        logger.info(f'mask bar {bar_num + 1} track {np.array(mask_tracks[i]) - 2}')

        if mask_diameter[i]:
            logger.info(f'mask bar {bar_num + 1} diameter')
            mask_tracks[i] = np.insert(mask_tracks[i],0,1)
        if mask_tensile[i]:
            logger.info(f'mask bar {bar_num + 1} tensile')
            mask_tracks[i] = np.insert(mask_tracks[i],0,0)



        track_pos_select_index = mask_tracks[i]

        # if not isinstance(track_pos_select_index,list):
        #     track_pos_select_index = [track_pos_select_index]



        # if mask_diameter[i]:
        #     logger.info(f'mask bar {bar_num + 1} diameter')
        #     track_pos_select_index = np.insert(track_pos_select_index,0,1)
        # if mask_tensile[i]:
        #     logger.info(f'mask bar {bar_num + 1} tensile')
        #     track_pos_select_index = np.insert(track_pos_select_index,0,0)



        for track_pos_index in track_pos_select_index:

            track_start_pos = track_with_bar_ctrl_pos[track_pos_index]
            if track_pos_index + 1 == len(track_positions):
                logger.info('why')
            track_end_pos = track_with_bar_ctrl_pos[track_pos_index + 1]
            # logger.info(track_start_pos)
            # logger.info(track_end_pos)
            masked_indices_pairs.append((track_start_pos, track_end_pos))
            # track_event = event[track_start_pos:track_end_pos]
            # logger.info(track_event)

    token_events = event.copy()

    for masked_pairs in masked_indices_pairs:
        masked_token = event[masked_pairs[0]:masked_pairs[1]]
        # logger.info(masked_token)
        decoder_in.append(vocab.mask_indices[0])
        for token in masked_token:
            decoder_in.append(vocab.char2index(token))
            decoder_target.append(vocab.char2index(token))
        else:
            decoder_target.append(vocab.eos_index)

    for masked_pairs in masked_indices_pairs[::-1]:
        # logger.info(masked_pairs)
        # logger.info(token_events[masked_pairs[0]:masked_pairs[1]])
        for pop_time in range(masked_pairs[1] - masked_pairs[0]):
            token_events.pop(masked_pairs[0])
        token_events.insert(masked_pairs[0], mask[0])

    for token in token_events:
        tokens.append(vocab.char2index(token))

    tokens = np.array(tokens)
    if len(decoder_in) > 0:
        decoder_in = np.array(decoder_in)
        decoder_target = np.array(decoder_target)
        # logger.info('\n')
        # logger.info(f'event length is {len(event)}')
        # logger.info(f'tokens length is {len(tokens)}')
        # logger.info(f'masked num is {masked_num}')
        # logger.info(f'decoder_in length is {len(decoder_in)}')
        # logger.info(f'decoder_out length is {len(decoder_target)}')
        # logger.info(f'ratio is {(len(tokens) + len(decoder_in)) / len(event)}')
        total_tokens.append(tokens)
        total_decoder_in.append(decoder_in)
        total_decoder_target.append(decoder_target)

    # logger.info(len(tokens) - len(np.where(output_label==2)[0]))
    # logger.info(len(output_label) - len(np.where(output_label==2)[0])*2)
    return torch.tensor(total_tokens).long().squeeze(), torch.tensor(total_decoder_target).long().squeeze(), mask_bars,mask_tracks

def model_generate(model, src, tgt,device,return_weights=False):

    src = src.clone().detach().unsqueeze(0).long().to(device)
    tgt = torch.tensor(tgt).unsqueeze(0).to(device)
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    tgt_mask = tgt_mask.clone().detach().unsqueeze(0).to(device)


    output,weights = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                           tgt_mask=tgt_mask)
    if return_weights:
        return output.squeeze(0).to('cpu'), weights.squeeze(0).to('cpu')
    else:
        return output.squeeze(0).to('cpu')

def get_args(default='.'):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--platform', default='local', type=str,
                        help="platform to run the code")
    parser.add_argument('-c', '--checkpoint_epoch', default=0, type=int,
                        help="checkpoint epoch number")
    parser.add_argument('-f', '--folder', default=None, type=str,
                        help="config folder")
    parser.add_argument('-g', '--generation', default=False, type=bool,
                        help="generation of bars or prediction of control token")
    return parser.parse_args()

def replace_bar_control(generated_output,generated_tensile_category,generated_diameter_category):
    tensile_idx = 0
    bar_pos = np.where(np.array(generated_output) == 'bar')[0]
    if len(generated_tensile_category) < len(bar_pos):
        generated_output = generated_output[:bar_pos[len(generated_tensile_category)]]

    for i,token in enumerate(generated_output):
        if token[:2] == 's_':
            generated_output[i] = 's_' + str(generated_tensile_category[tensile_idx])
            generated_output[i+1] = 'a_' + str(generated_diameter_category[tensile_idx])
            tensile_idx += 1

    return generated_output

def replace_track_control(generated_output,
                      generated_track_control,key_name):

    key = key_to_token[key_name]

    generated_output[3] = key
    r = re.compile('i_\d')

    track_program = list(filter(r.match, generated_output))
    track_num_pos = np.where(track_program[0] == np.array(generated_output))[0][0]
    for i,token in enumerate(generated_output[4:track_num_pos]):
        generated_output[4+i] = generated_track_control[i]
    return generated_output



def main():
    args = get_args()


    platform = args.platform
    checkpoint_epoch = args.checkpoint_epoch
    config_folder = args.folder
    is_generation = args.generation

    logger.info(f'platform is {platform}')
    logger.info(f'checkpoint epoch is {checkpoint_epoch}')
    logger.info(f'folder is {config_folder}')
    if is_generation:
        logger.info('generate masked tracks')
    else:
        logger.info('predict masked control tokens')


    vocab = WordVocab(all_tokens)
    with open(os.path.join(config_folder,"files/config.yaml")) as file:

        config = yaml.full_load(file)


    model = ScoreTransformer(vocab.vocab_size, config['d_model']['value'], config['nhead']['value'], config['num_encoder_layers']['value'],
                                 config['num_encoder_layers']['value'], 2048, 2400,
                                 0.1, 0.1)

    checkpoint = os.path.join(config_folder,f"files/checkpoint_{checkpoint_epoch}")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model_dict = torch.load(checkpoint,map_location=device)

    model_state = model_dict['model_state_dict']
    # optimizer_state = model_dict['optimizer_state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    # new_state_dict = model_state

    model.to(device)

    model.load_state_dict(new_state_dict)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)



    window_size = int(16 / 2)

    if platform == 'local':
        folder_prefix = '/home/ruiguo/'
    else:
        folder_prefix = '/content/drive/MyDrive/'


    test_batch_name = 'test_batches_0_0_1'
    test_length_name = 'test_batch_lengths_0_0_1'

    # test_batch_name = 'train_batches_0_8'
    # test_length_name = 'train_batch_lengths_0_8'


    test_batches = pickle.load(open(folder_prefix + 'score_transformer/data/' + test_batch_name, 'rb'))

    test_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/data/' + test_length_name, 'rb'))

    # selected_indices = np.sort(np.random.choice(range(4900,len(test_batches)),5000))
    # for i in range(len(test_batches)-1, 0, -1):
    #     if i not in selected_indices:
    #         del test_batches[i]


    logger.info(f'test batch length is {len(test_batches)}')

    mask_bars = None
    mask_tracks = [2]
    mask_tensile = [0]
    mask_diameter = [0]
    diameter_control_change = 0
    tensile_control_change = 0
    track_control_change = 5
    total_num = 0
    new_data  = []
    for index0,batches in enumerate(test_batches):
        for index1,batch in enumerate(batches):
            logger.info(f'the {total_num} data')

            total_num += 1
            # if total_num != 13519:
            #     continue

            logger.info(f'original track control is {batch[4:16]}')

            if is_generation:

                r = re.compile('i_\d')

                track_program = list(filter(r.match, batch))
                track_nums = len(track_program)

                # if change track control, mask a whole track
                # if change bar control, random choose several bars
                # and mask random tracks in that bar

                if np.random.rand() > 0.01:
                    mask_track = True
                    # change track control
                    # random select one track
                    # logger.info('change track control')
                    selected_track = np.random.choice(track_nums)
                    mask_tracks = [selected_track]
                    mask_bars = None

                    selected_control_name = np.random.choice(vocab.track_control_names)

                    track_num_pos = np.where(track_program[0] == np.array(batch))[0][0]
                    original_track_tokens = batch[4:track_num_pos]
                    curr_track = 0
                    for i, token in enumerate(original_track_tokens):
                        if vocab.token_class_ranges[vocab.char2index(token)] == selected_control_name:
                            if curr_track == selected_track:
                                # change token
                                original_track_control = original_track_tokens[i]
                                new_track_control = np.random.choice(vocab.name_to_tokens[selected_control_name])
                                batch[4+i] = new_track_control
                                logger.info(f'change track {selected_track} control from {original_track_control} to {new_track_control}')
                                break
                            curr_track += 1
                else:
                    mask_track = False
                    # select number of bar control to change
                    # logger.info('change bar control')
                    bar_poses = np.where(np.array(batch) == 'bar')[0]
                    # mask less than 8 bars to make it faster
                    mask_bar_nums = np.random.choice(8) + 1
                    if mask_bar_nums > len(bar_poses):
                        mask_bar_nums = len(bar_poses)

                    mask_bars = np.sort(np.random.choice(len(bar_poses),mask_bar_nums,replace = False))
                    # selected_bar_poses = bar_poses[mask_bars]
                    mask_bars = mask_bars.tolist()
                    mask_tracks = None
                    mask_tensile = [0]
                    mask_diameter = [0]

                    for bar in mask_bars:
                        bar_pos = bar_poses[bar]
                        if vocab.token_class_ranges[vocab.char2index(batch[bar_pos+1])] in vocab.bar_control_names:
                            # change tensile or diameter
                            if np.random.rand() > 0.5:
                                original_tensile = batch[bar_pos + 1]
                                batch[bar_pos + 1] = np.random.choice(vocab.name_to_tokens['tensile'])
                                logger.info(f'change bar {bar+1} tensile from {original_tensile} to {batch[bar_pos + 1]}')
                            else:
                                original_diameter = batch[bar_pos + 2]
                                batch[bar_pos + 2] = np.random.choice(vocab.name_to_tokens['diameter'])
                                logger.info(f'change bar {bar+1} diameter from {original_diameter} to {batch[bar_pos + 2]}')

                        else:
                            index = 0
                            while vocab.token_class_ranges[vocab.char2index(batch[bar_pos+index])] not in vocab.bar_control_names:
                                index += 1
                            if np.random.rand() > 0.5:
                                original_tensile = batch[bar_pos + index]
                                batch[bar_pos+index] = np.random.choice(vocab.name_to_tokens['tensile'])
                                logger.info(f'change bar {bar+1} tensile from {original_tensile} to {batch[bar_pos+index]}')
                            else:
                                original_diameter = batch[bar_pos + index + 1]
                                batch[bar_pos+index+1] = np.random.choice(vocab.name_to_tokens['diameter'])
                                logger.info(f'change bar {bar+1} diameter from {original_diameter} to {batch[bar_pos+index+1]}')

                result = generate(model, batch, mask_bars, mask_tracks,mask_tensile,mask_diameter, device)
                if result is None:
                    logger.info(f'skip batches {index0} batch {index1}]')
                    continue
                restored_token, mask_bars = result
                restored_with_generated_token, restored_with_target_token, restored_empty_fill = restored_token


                restored_with_target_pm,_ = event_2midi(restored_with_target_token.tolist())
                restored_with_target_pm.write(f'restore_target_{total_num-1}.mid')


                restored_with_generated_pm,_ = event_2midi(restored_with_generated_token.tolist())
                restored_with_generated_pm.write(f'restore_generated_{total_num-1}.mid')

                # restored_with_empty_pm,_ = event_2midi(restored_empty_fill.tolist())
                # restored_with_empty_pm.write(f'restore_empty_{index+index1}.mid')



                result = cal_bar_control(f'restore_target_{total_num-1}.mid', f'restore_generated_{total_num-1}.mid')
                if result is None:
                    logger.info(f'skip batches {index0} batch {index1}]')
                    continue
                generated_tensile_category, generated_diameter_category, target_tensile_category, target_diameter_category, key_name_generated, key_name_target = result
                track_num_pos = np.where(track_program[0] == np.array(restored_with_generated_token))[0][0]
                bar_poses = np.where(np.array(restored_with_generated_token) == 'bar')[0]
                generated_bar_number = len(bar_poses)
                if len(generated_tensile_category) < generated_bar_number:
                    restored_with_generated_token = restored_with_generated_token[:bar_poses[len(generated_tensile_category)]]
                    logger.info('change generated bar length to match tension calculation')

                r = re.compile('s_\d')

                target_tensile_category_in_file = list(filter(r.match, restored_with_generated_token))

                r = re.compile('a_\d')

                target_diameter_category_in_file = list(filter(r.match, restored_with_generated_token))

                target_track_control_in_file = restored_with_generated_token[4:track_num_pos]
                if not mask_track:
                    for bar in mask_bars:
                        if bar >= len(generated_tensile_category):
                            logger.info(f'bar {bar} out of range of total bar length {len(generated_tensile_category)} now')
                        else:
                            logger.info(f'bar {bar+1}, target tensile is {target_tensile_category_in_file[bar]} , generated tensile is {generated_tensile_category[bar]} \n'
                                  f'bar {bar+1}, target diameter is {target_diameter_category_in_file[bar]} , generated diameter is {generated_diameter_category[bar]} \n')
                            if abs(int(target_tensile_category_in_file[bar][2:]) - generated_tensile_category[bar]) > 0 or abs(int(target_diameter_category_in_file[bar][2:]) - generated_diameter_category[bar]) > 0:
                                # if len(generated_tensile_category) == len(generated_diameter_category) == total_bars:
                                logger.info(f'bar control does not match, add to new data')
                                data_with_true_bar_control = replace_bar_control(restored_with_generated_token,
                                                                                 generated_tensile_category,
                                                                                 generated_diameter_category)

                                generated_track_control = cal_track_control(data_with_true_bar_control,
                                                                            restored_with_generated_pm)

                                data_with_true_track_control = replace_track_control(data_with_true_bar_control,
                                                                                     generated_track_control,
                                                                                     key_name_generated)

                                new_data.append(data_with_true_track_control.tolist())

                                break

                else:
                    data_with_true_bar_control = replace_bar_control(restored_with_generated_token,
                                                                     generated_tensile_category,
                                                                     generated_diameter_category)

                    generated_track_control = cal_track_control(data_with_true_bar_control,
                                                                restored_with_generated_pm)

                    # for i, target_control in enumerate(batch[4:track_num_pos]):
                    for i, target_control in enumerate(target_track_control_in_file):

                        logger.info(
                            f'target track control is {target_control} , generated track control is {generated_track_control[i]} \n')
                        if abs(int(target_control[2:]) - int(generated_track_control[i][2:])) > 0:
                            logger.info('track control does not match, add to new data')
                            data_with_true_track_control = replace_track_control(data_with_true_bar_control,
                                                                                 generated_track_control,
                                                                                 key_name_generated)

                            new_data.append(data_with_true_track_control)

                            break


                # logger.info('generated track control')


                # original_pm, _ = event_2midi(batch)
                # original_pm.write(f'original_{index+index1}.mid')
                #
                # # cal_track_control(np.array(batch), original_pm)
                #
                # if diameter_control_change != 0:
                #     logger.info('change diameter control')
                #     batch = change_control(batch, diameter_control_change, diameter_token,0,11,mask_bars=mask_bars)
                #
                #     (new_bar_control_restored_with_generated_token, new_bar_control_restored_with_target_token, new_bar_control_restored_empty_fill),mask_bars = generate(model, batch,
                #                                                                                               mask_bars,
                #                                                                                               mask_tracks,
                #                                                                                         mask_tensile=[True],
                #                                                                                         mask_diameter=[False],
                #                                                                                         device=device)
                #
                #     new_bar_control_restored_with_target_pm, _ = event_2midi(new_bar_control_restored_with_target_token.tolist())
                #     new_bar_control_restored_with_target_pm.write(f'new_bar_diameter_control_restore_target_{index+index1}.mid')
                #
                #     new_bar_control_restored_with_generated_pm, _ = event_2midi(new_bar_control_restored_with_generated_token.tolist())
                #     new_bar_control_restored_with_generated_pm.write(f'new_bar_diameter_control_restore_generated_{index+index1}.mid')
                #
                #     new_bar_control_restored_with_empty_pm, _ = event_2midi(new_bar_control_restored_empty_fill.tolist())
                #     new_bar_control_restored_with_empty_pm.write(f'new_bar_diameter_control_restore_empty_{index+index1}.mid')
                #
                #     generated_tensile_category, generated_diameter_category,target_tensile_category,target_diameter_category,key_name_generated,key_name_target = cal_bar_control(f'new_bar_diameter_control_restore_target_{index+index1}.mid', f'new_bar_diameter_control_restore_generated_{index+index1}.mid')
                #
                #     for bar in mask_bars:
                #         if bar < len(generated_tensile_category) and bar < len(generated_diameter_category)  and bar < len(target_diameter_category) and bar < len(target_tensile_category):
                #             logger.info(
                #                 f'bar {bar+1}, original tensile is {target_tensile_category[bar]} , generated tensile is {generated_tensile_category[bar]} \n'
                #                 f'bar {bar+1}, original diameter is {target_diameter_category[bar]}, target diameter is {target_diameter_category[bar]  + diameter_control_change} , generated diameter is {generated_diameter_category[bar]} \n')
                #
                #     logger.info('generated track control')
                #     cal_track_control(new_bar_control_restored_with_generated_token,
                #                       new_bar_control_restored_with_generated_pm)
                #     logger.info('original track control')
                #     cal_track_control(new_bar_control_restored_with_target_token,
                #                       new_bar_control_restored_with_target_pm)
                #
                # if tensile_control_change != 0:
                #     logger.info('change tensile control')
                #     batch = change_control(batch, tensile_control_change, tensile_strain_token, 0, 11, mask_bars=mask_bars)
                #
                #     (new_bar_control_restored_with_generated_token, new_bar_control_restored_with_target_token, new_bar_control_restored_empty_fill),mask_bars = generate(
                #         model, batch,
                #         mask_bars,
                #         mask_tracks,
                #         mask_tensile=[False],
                #         mask_diameter=[True],
                #         device=device)
                #
                #     new_bar_control_restored_with_target_pm, _ = event_2midi(
                #         new_bar_control_restored_with_target_token.tolist())
                #     new_bar_control_restored_with_target_pm.write(f'new_bar_tensile_control_restore_target_{index+index1}.mid')
                #
                #     new_bar_control_restored_with_generated_pm, _ = event_2midi(
                #         new_bar_control_restored_with_generated_token.tolist())
                #     new_bar_control_restored_with_generated_pm.write(f'new_bar_tensile_control_restore_generated_{index+index1}.mid')
                #
                #     new_bar_control_restored_with_empty_pm, _ = event_2midi(
                #         new_bar_control_restored_empty_fill.tolist())
                #     new_bar_control_restored_with_empty_pm.write(f'new_bar_tensile_control_restore_empty_{index+index1}.mid')
                #
                #     generated_tensile_category, generated_tensile_category, target_tensile_category, target_tensile_category,key_name_generated,key_name_target = cal_bar_control(
                #         f'new_bar_tensile_control_restore_target_{index+index1}.mid', f'new_bar_tensile_control_restore_generated_{index+index1}.mid')
                #
                #     for bar in mask_bars:
                #         if bar < len(generated_tensile_category) and bar < len(generated_diameter_category)  and bar < len(target_diameter_category) and bar < len(target_tensile_category):
                #             logger.info(
                #                 f'bar {bar+1}, original tensile is {target_tensile_category[bar]}, target tensile is {target_tensile_category[bar] + tensile_control_change} , generated tensile is {generated_tensile_category[bar]} \n'
                #                 f'bar {bar+1}, original diameter is {target_diameter_category[bar]} , generated diameter is {generated_diameter_category[bar]} \n')
                #
                #     # bar control change should not change track control
                #     # logger.info('generated track token')
                #     # logger.info(new_bar_control_restored_with_generated_token)
                #     logger.info('generated track control')
                #     cal_track_control(new_bar_control_restored_with_generated_token, new_bar_control_restored_with_generated_pm)
                #     logger.info('original track control')
                #     cal_track_control(new_bar_control_restored_with_target_token, new_bar_control_restored_with_target_pm)
                #
                # if track_control_change != 0:
                #     logger.info('change track control')
                #     batch = change_control(batch, track_control_change, track_pitch_register_token,0,7,mask_bars=mask_bars,mask_tracks=mask_tracks)
                #
                #     (new_track_control_restored_with_generated_token, new_track_control_restored_with_target_token, new_track_control_restored_empty_fill),mask_bars = generate(model, batch,
                #                                                                                               mask_bars,
                #                                                                                               mask_tracks,
                #                                                                                             mask_tensile=[False],
                #                                                                                             mask_diameter=[False],
                #                                                                                             device=device)
                #
                #
                #     new_track_control_restored_with_target_pm, _ = event_2midi(new_track_control_restored_with_target_token.tolist())
                #     new_track_control_restored_with_target_pm.write(f'new_track_control_restore_target_{index+index1}.mid')
                #
                #     new_track_control_restored_with_generated_pm, _ = event_2midi(new_track_control_restored_with_generated_token.tolist())
                #     new_track_control_restored_with_generated_pm.write(f'new_track_control_restore_generated_{index+index1}.mid')
                #
                #     new_track_control_restored_with_empty_pm, _ = event_2midi(new_track_control_restored_empty_fill.tolist())
                #     new_track_control_restored_with_empty_pm.write(f'new_track_control_restore_empty_{index+index1}.mid')
                #
                #     generated_tensile_category, generated_diameter_category,target_tensile_category,target_diameter_category,key_name_generated,key_name_target = cal_bar_control(f'new_track_control_restore_target_{index+index1}.mid', f'new_track_control_restore_generated_{index+index1}.mid')
                #
                #     # track control change should not change bar control ?
                #     for bar in mask_bars:
                #         if bar < len(generated_tensile_category) and bar < len(generated_diameter_category)  and bar < len(target_diameter_category) and bar < len(target_tensile_category):
                #             logger.info(
                #                 f'bar {bar}, original tensile is {target_tensile_category[bar]} , generated tensile is {generated_tensile_category[bar]} \n'
                #                 f'bar {bar}, original diameter is {target_diameter_category[bar]} , generated diameter is {generated_diameter_category[bar]} \n')
                #
                #     # logger.info('generated track token')
                #     # logger.info(new_track_control_restored_with_generated_token)
                #
                #     cal_track_control(new_track_control_restored_with_generated_token, new_track_control_restored_with_generated_pm)
                #     logger.info('target track token')
                #     # cal_track_control(new_track_control_restored_with_target_token, new_track_control_restored_with_target_pm)
                #     logger.info(batch[4:16])
            else:
                src_token, generated_output, target_output, tgt_inp_token,weight_list = prediction(model, batch,['tensile'],device)

    pickle.dump(new_data, open('new_data', 'wb'))

            # logger.info(0)





main()
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


    file_events = preprocessing.remove_control_event(file_events.tolist())
    r = re.compile('i_\d')

    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)



    file_events = np.array(file_events)

    #     print(f'number of bars is {len(bar_pos)}')
    #     print(f'time signature is {file_event[1]}')
    bar_length = int(file_events[1][0])
    bar_pos = np.where(file_events == 'bar')[0]
    if bar_length != 6:
        bar_length = bar_length * 4 * len(bar_pos)
    else:
        bar_length = bar_length / 2 * 4 * len(bar_pos)
    #     print(f'bar length is {bar_length}')

    track_events = {}

    for i in range(num_of_tracks):
        track_events[f'track_{i}'] = []
    track_names = list(track_events.keys())
    for bar_index in range(len(bar_pos) - 1):
        bar = bar_pos[bar_index]
        next_bar = bar_pos[bar_index + 1]
        bar_events = file_events[bar:next_bar]
        #         print(bar_events)

        track_pos = []

        for track_name in track_names:
            if len(np.where(track_name == bar_events)[0]) == 0:
                print(bar_events)
            track_pos.append(np.where(track_name == bar_events)[0][0])
        #         print(track_pos)
        for track_index in range(len(track_names) - 1):
            track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
            track_events[track_names[track_index]].append(track_event)
        #             print(track_event)
        else:
            track_index += 1
            track_event = bar_events[track_pos[track_index]:]
            #             print(track_event)
            track_events[track_names[track_index]].append(track_event)

    densities = dataset.note_density(track_events, bar_length)
    density_category = dataset.to_category(densities, dataset.control_bins)

    occupation_rate, polyphony_rate = dataset.occupation_polyphony_rate(pm)
    occupation_category = dataset.to_category(occupation_rate, dataset.control_bins)
    polyphony_category = dataset.to_category(polyphony_rate, dataset.control_bins)
    pitch_register_category = dataset.pitch_register(track_events)
    #     print(densities)
    #     print(occupation_rate)
    #     print(polyphony_rate)
    #     print(density_category)
    #     print(occupation_category)
    #     print(polyphony_category)

    #     key_token =  key_to_token[key]

    density_token = [f'd_{category}' for category in density_category]
    occupation_token = [f'o_{category}' for category in occupation_category]
    polyphony_token = [f'y_{category}' for category in polyphony_category]
    pitch_register_token = [f'r_{category}' for category in pitch_register_category]

    track_control_tokens = density_token + occupation_token + polyphony_token + pitch_register_token

    print(track_control_tokens)







def cal_bar_control(target_name, generated_name):


    result = tension_calculation.extract_notes(target_name, 3)


    pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result

    key_name = tension_calculation.all_key_names

    result_target = tension_calculation.cal_tension(
        target_name, piano_roll, sixteenth_time, beat_time, beat_indices, down_beat_time,
        down_beat_indices, './', 1, key_name)

    total_tension_target, diameters_target, _, key_name_target, key_change_time_target, key_change_bar_target, key_change_name_target = result_target

    target_tensile_category = dataset.to_category(total_tension_target, dataset.tensile_bins)
    target_diameter_category = dataset.to_category(diameters_target, dataset.tensile_bins)

    result = tension_calculation.extract_notes(generated_name, 3)

    pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result

    key_name = tension_calculation.all_key_names

    result_generated = tension_calculation.cal_tension(
        target_name, piano_roll, sixteenth_time, beat_time, beat_indices, down_beat_time,
        down_beat_indices, './', 1, key_name)

    total_tension_generated, diameters_generated, _, key_name_generated, key_change_time_generated, key_change_bar_generated, key_change_name_generated = result_generated

    generated_tensile_category = dataset.to_category(total_tension_generated, dataset.tensile_bins)
    generated_diameter_category = dataset.to_category(diameters_generated, dataset.tensile_bins)



    print(f'target key {key_name_target}, generated key {key_name_generated}')

    return generated_tensile_category, generated_diameter_category,target_tensile_category,target_diameter_category


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
                output = model_generate(model, src, tgt_inp + this_tgt_inp).to(device)
                index = sampling(output[-1], 0.9)
                sampling_times += 1

                event = vocab.index2char(index)
                # print(event)

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
                                output = model_generate(model, src, tgt_inp + this_tgt_inp).to(device)
                                index = sampling(output[-1], 0.9)
                                event = vocab.index2char(index)
                                sampling_times += 1
                                if mask_track_name in ['s_', 'a_']:
                                    compare_event = event[:2]
                                else:
                                    compare_event = event
                            if compare_event != mask_track_name:
                                print(f'{compare_event} is not equal to {mask_track_name}')
                                print(f'mask {mask_idx} needs to be generated again')
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
                                output = model_generate(model, src, tgt_inp + this_tgt_inp).to(device)
                                index = sampling(output[-1], 0.9)
                                event = vocab.index2char(index)
                                sampling_times += 1
                            if event != pitches:
                                print(f'{event} is not pitch')
                                print(f'mask {mask_idx} needs to be generated again')
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
                            print(f'{curr_time} is not equal to {bar_duration}')
                            print(f'mask {mask_idx} needs to be generated again')
                            this_tgt_failure = True



                # while not bar_track_validate(this_tgt_inp,index,mask_track_name,bar_duration) and sampling_times < 10:
                #
                #     index = sampling(output[-1], 0.9)
                #     sampling_times += 1
                #     print(f'sampling times {sampling_times}')
                #
                # if not bar_track_validate(this_tgt_inp,index,mask_track_name,bar_duration):
                #     output = model_generate(model, src, this_tgt_inp).to(device)
                #     sampling_times = 0
                #     index = sampling(output[-1], 0.9)
                #     sampling_times += 1
                #     while not bar_track_validate(this_tgt_inp, index, mask_track_name,bar_duration) and sampling_times < 10:
                #         index = sampling(output[-1], 0.9)
                #         sampling_times += 1
                #     if not bar_track_validate(this_tgt_inp,index,mask_idx,mask_bars,mask_tracks,mask_bar_control):
                #         print('cannot generate valid data after trying twice')
                #         return None

                this_tgt_inp.append(index)
            if this_tgt_inp[-1] == vocab.char2index('<eos>') and not this_tgt_failure:
                mask_idx += 1
                tgt_inp.extend(this_tgt_inp)
            else:
                print(f'generate again, generate time is {generate_times}')
                generate_times += 1
                if generate_times > 5:
                    print(f'fail to generate track {mask_idx} after 5 times')
                    tgt_inp.extend(this_tgt_inp)
                    mask_idx += 1








                # i += 1
    generated_output = []
    target_output = []
    src_token = []

    for i, token_idx in enumerate(tgt_inp):
        output_token = vocab.index2char(token_idx)
        generated_output.append(output_token)

    for i, token_idx in enumerate(tgt_out):
        target_token = vocab.index2char(token_idx.item())
        target_output.append(target_token)

    # print(f'generation {generated_output}')
    # print(f'target {target_output}')

    for i, token_idx in enumerate(src):
        src_token.append(vocab.index2char(token_idx.item()))
    return restore_marked_input(src_token,generated_output,target_output)



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


def get_args(default='.'):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--platform', default='local', type=str,
                        help="platform to run the code")
    parser.add_argument('-c', '--checkpoint_epoch', default=0, type=int,
                        help="checkpoint epoch number")
    parser.add_argument('-f', '--folder', default=None, type=str,
                        help="config folder")
    return parser.parse_args()


def restore_marked_input(src_token, generated_output, target_output):
    src_token = np.array(src_token)

    # restore with generated output
    restored_with_generated_token = src_token.copy()

    generated_output = np.array(generated_output)
    target_output = np.array(target_output)
    generation_mask_indices = np.where(generated_output == 'm_0')[0]

    if len(generation_mask_indices) == 1:

        mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
        generated_result_sec = generated_output[generation_mask_indices[0] + 1:]

        #         print(len(generated_result_sec))
        restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
        for token in generated_result_sec[-2::-1]:
            #             print(token)
            restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)


    else:

        for i in range(len(generation_mask_indices) - 1):
            #         print(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i] + 1:generation_mask_indices[i + 1]]

            #             print(len(generated_result_sec))
            #             print(mask_indices[i])
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])

            for token in generated_result_sec[-2::-1]:
                #                 print(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)

        else:
            #         print(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i + 1] + 1:]

            #             print(len(generated_result_sec))
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
            for token in generated_result_sec[-2::-1]:
                #                 print(token)
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

    for name in target_mask_names:
        result_pos = target_output.index(name, start_index, end_index)
        target_mask_indices.append(result_pos)
        start_index = result_pos
    # print(target_mask_indices)
    restored_with_target_token = src_token.copy()

    if len(target_mask_indices) == 1:
        mask_indices = np.where(restored_with_target_token == 'm_0')[0]
        target_result_sec = target_output[target_mask_indices[0]:end_index - 1]

        #         print(len(target_result_sec))
        restored_with_target_token = np.delete(restored_with_target_token, mask_indices[0])
        for token in target_result_sec[::-1]:
            #             print(token)
            restored_with_target_token = np.insert(restored_with_target_token, mask_indices[0], token)

    else:
        for i in range(len(target_mask_indices) - 1):

            mask_indices = np.where(restored_with_target_token == 'm_0')[0]
            target_result_sec = target_output[target_mask_indices[i]:target_mask_indices[i + 1] - 1]

            #             print(len(target_result_sec))
            #             print(mask_indices[i])
            restored_with_target_token = np.delete(restored_with_target_token, mask_indices[0])
            for token in target_result_sec[::-1]:
                #                 print(token)
                restored_with_target_token = np.insert(restored_with_target_token, mask_indices[0], token)

        else:
            mask_indices = np.where(restored_with_target_token == 'm_0')[0]
            target_result_sec = target_output[target_mask_indices[i + 1]:end_index - 1]

            #             print(len(target_result_sec))
            restored_with_target_token = np.delete(restored_with_target_token, mask_indices[0])
            for token in target_result_sec[::-1]:
                #                 print(token)
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

        # print(target_mask_names)
        mask_indices = np.where(restored_empty_fill == 'm_0')[0]


        # print(mask_indices)
        restored_empty_fill = np.delete(restored_empty_fill, mask_indices[0])
        if target_mask_names[i][0] == 't':
            for token in fill_in_tokens[::-1]:
                # print(token)
                restored_empty_fill = np.insert(restored_empty_fill, mask_indices[0], token)
                # print(target_mask_names[i])
        restored_empty_fill = np.insert(restored_empty_fill, mask_indices[0], target_mask_names[i])



    # for i, index in enumerate(mask_indices):
    #     if index > bar_pos[-1]:
    #         print(f'masked bar {len(bar_pos)} {track_names[i]}')
    #     else:
    #         bar_start_index = np.where(index < bar_pos)[0][0]
    #         print(f'masked bar {bar_start_index} {track_names[i]}')

    return restored_with_generated_token, restored_with_target_token, restored_empty_fill


def change_control(batch, control_change, control_tokens,mininum,maximum,mask_bars=None):
    new_tokens = []
    if mask_bars:
        cur_bar = 0
        counter = 0
        for token in batch:
            if token in control_tokens:
                counter += 1
                if cur_bar in mask_bars:
                    token_new_category = int(token[2:]) + control_change
                    if token_new_category < mininum:
                        token_new_category = 0
                    if token_new_category > maximum:
                        token_new_category = maximum
                    token_new_category = str(token_new_category)
                    token = token[:2] + token_new_category
                if counter % 2 == 0:
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

    return new_tokens


# mask song/track/bar control token, span=1 for all
def mask_category(events, token_type):

    total_tokens = []
    total_decoder_in = []
    total_decoder_target = []
    
    for event in events:
        tokens = []
        decoder_in = []
        decoder_target = []
        start_pos = 0
        total_masked_ratio = 0
        masked_num = 0
        while start_pos < len(event):


            # add track selection
            # if event[start_pos] == 'track_0':
            #     current_track = 0
            #     current_track_unmasked = True
            # if event[start_pos] == 'track_1':
            #     current_track = 1
            #     current_track_unmasked = True
            # if event[start_pos] == 'track_2':
            #     current_track = 2
            #     current_track_unmasked = True

            if vocab.token_class_ranges[vocab.char2index(event[start_pos])] == token_type:
                masked_token = [event[start_pos]]
                tokens.append(vocab.mask_indices[0])

                decoder_in.append(vocab.mask_indices[0])
                decoder_in.append(vocab.char2index(masked_token))
                decoder_target.append(vocab.char2index(masked_token))
                decoder_target.append(vocab.eos_index)

                masked_num += 1
                total_masked_ratio += 1 / len(event)
                start_pos += 1

            else:
                tokens.append(vocab.char2index(event[start_pos]))
                start_pos += 1
        tokens = np.array(tokens)

        if len(decoder_in) > 0:
            decoder_in = np.array(decoder_in)
            decoder_target = np.array(decoder_target)
            # print('\n')
            # print(f'event length is {len(event)}')
            # print(f'tokens length is {len(tokens)}')
            # print(f'masked num is {masked_num}')
            # print(f'decoder_in length is {len(decoder_in)}')
            # print(f'decoder_out length is {len(decoder_target)}')
            # print(f'ratio is {(len(tokens) + len(decoder_in)) / len(event)}')
            total_tokens.append(tokens)
            total_decoder_in.append(decoder_in)
            total_decoder_target.append(decoder_target)

    # print(len(tokens) - len(np.where(output_label==2)[0]))
    # print(len(output_label) - len(np.where(output_label==2)[0])*2)
    return total_tokens, total_decoder_in, total_decoder_target



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

    #     print(f'bar number prob is {bar_number_prob}')
    #     bar_number_prob /= np.sum(bar_number_prob)

    #     # if np.sum(bar_number_prob) != 1:
    #     #     print(bar_number_prob)
    #     # print(f'bar number prob is {bar_number_prob}')
    #     resample_counts = np.random.multinomial(1, bar_number_prob[:len(bar_pos)])
    #     bar_mask_number = np.where(resample_counts)[0][0] + 1
    #     # bar_mask_number = np.random.choice(range(len(bar_pos)), size=1, p=bar_ratio[:len(bar_pos)])
    #     bar_start_indices = np.sort(np.random.choice(len(bar_pos),size=bar_mask_number,replace=False))

    if mask_bars is None:
        mask_bars = range(len(bar_pos))
        print(mask_bars)
    else:
        if not isinstance(mask_bars,list):
            if mask_bars >= len(bar_pos):
                print(f'mask bar {mask_bars} is larger than total bar {len(bar_pos)}, mask bar 1 instead')
                mask_bars  = 1
            mask_bars = [mask_bars]
        else:
            if len(mask_bars) >= len(bar_pos):
                mask_bars = mask_bars[0:len(bar_pos)]

    for i, bar in enumerate(mask_bars):
        if bar > len(bar_pos) - 1:
            print(f'mask bar {bar} is larger than total bar {len(bar_pos)}, mask bar {i} instead')
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



    # print(f'bar start indices is {bar_start_indices}')
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
        #         # print(prob)
        #         if prob < ratios[0]:
        #             # select one track
        #             track_pos_select_index = [np.random.choice(track_num)]
        #         elif prob < ratios[0] + ratios[1]:
        #             # select two tracks
        #             track_pos_select_index = np.sort(np.random.choice(track_num, 2, replace=False))
        #         else:
        #             # select three tracks
        #             track_pos_select_index = np.arange(track_num)
        # print(track_nums)
        # print(bar_start_index)


        print(f'mask bar {bar_num + 1} track {np.array(mask_tracks[i]) - 2}')

        if mask_diameter[i]:
            print(f'mask bar {bar_num + 1} diameter')
            mask_tracks[i] = np.insert(mask_tracks[i],0,1)
        if mask_tensile[i]:
            print(f'mask bar {bar_num + 1} tensile')
            mask_tracks[i] = np.insert(mask_tracks[i],0,0)



        track_pos_select_index = mask_tracks[i]

        # if not isinstance(track_pos_select_index,list):
        #     track_pos_select_index = [track_pos_select_index]



        # if mask_diameter[i]:
        #     print(f'mask bar {bar_num + 1} diameter')
        #     track_pos_select_index = np.insert(track_pos_select_index,0,1)
        # if mask_tensile[i]:
        #     print(f'mask bar {bar_num + 1} tensile')
        #     track_pos_select_index = np.insert(track_pos_select_index,0,0)



        for track_pos_index in track_pos_select_index:

            track_start_pos = track_with_bar_ctrl_pos[track_pos_index]
            if track_pos_index + 1 == len(track_positions):
                print('why')
            track_end_pos = track_with_bar_ctrl_pos[track_pos_index + 1]
            # print(track_start_pos)
            # print(track_end_pos)
            masked_indices_pairs.append((track_start_pos, track_end_pos))
            # track_event = event[track_start_pos:track_end_pos]
            # print(track_event)

    token_events = event.copy()

    for masked_pairs in masked_indices_pairs:
        masked_token = event[masked_pairs[0]:masked_pairs[1]]
        # print(masked_token)
        decoder_in.append(vocab.mask_indices[0])
        for token in masked_token:
            decoder_in.append(vocab.char2index(token))
            decoder_target.append(vocab.char2index(token))
        else:
            decoder_target.append(vocab.eos_index)

    for masked_pairs in masked_indices_pairs[::-1]:
        # print(masked_pairs)
        # print(token_events[masked_pairs[0]:masked_pairs[1]])
        for pop_time in range(masked_pairs[1] - masked_pairs[0]):
            token_events.pop(masked_pairs[0])
        token_events.insert(masked_pairs[0], mask[0])

    for token in token_events:
        tokens.append(vocab.char2index(token))

    tokens = np.array(tokens)
    if len(decoder_in) > 0:
        decoder_in = np.array(decoder_in)
        decoder_target = np.array(decoder_target)
        # print('\n')
        # print(f'event length is {len(event)}')
        # print(f'tokens length is {len(tokens)}')
        # print(f'masked num is {masked_num}')
        # print(f'decoder_in length is {len(decoder_in)}')
        # print(f'decoder_out length is {len(decoder_target)}')
        # print(f'ratio is {(len(tokens) + len(decoder_in)) / len(event)}')
        total_tokens.append(tokens)
        total_decoder_in.append(decoder_in)
        total_decoder_target.append(decoder_target)

    # print(len(tokens) - len(np.where(output_label==2)[0]))
    # print(len(output_label) - len(np.where(output_label==2)[0])*2)
    return torch.tensor(total_tokens).long().squeeze(), torch.tensor(total_decoder_target).long().squeeze(), mask_bars,mask_tracks


def main():
    args = get_args()

    platform = args.platform
    checkpoint_epoch = args.checkpoint_epoch
    config_folder = args.folder

    print(f'platform is {platform}')
    print(f'checkpoint epoch is {checkpoint_epoch}')
    print(f'folder is {config_folder}')


    vocab = WordVocab(all_tokens)
    with open(os.path.join(config_folder,"files/config.yaml")) as file:

        config = yaml.full_load(file)


    model = ScoreTransformer(vocab.vocab_size, config['d_model']['value'], config['nhead']['value'], config['num_encoder_layers']['value'],
                                 config['num_encoder_layers']['value'], 2048, 2400,
                                 0.1, 0.1)

    checkpoint = os.path.join(config_folder,f"files/checkpoint_{checkpoint_epoch}")
    model_dict = torch.load(checkpoint)

    model_state = model_dict['model_state_dict']
    optimizer_state = model_dict['optimizer_state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    # new_state_dict = model_state

    model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    window_size = int(16 / 2)

    if platform == 'local':
        folder_prefix = '/home/ruiguo/'
    else:
        folder_prefix = '/content/drive/MyDrive/'


    test_batch_name = 'test_batches_0_0_1_new_bins'
    test_length_name = 'test_batch_lengths_0_0_1_new_bins'


    test_batches = pickle.load(open(folder_prefix + 'score_transformer/' + test_batch_name, 'rb'))

    test_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/' + test_length_name, 'rb'))


    print(f'test batch length is {len(test_batches)}')

    mask_bars = [0,2,3,4,5,7]
    mask_tracks = [0,1]
    mask_tensile = [0,1,0,1,1,1]
    mask_diameter = [1]
    diameter_control_change = 4
    tensile_control_change = 2
    track_control_change = 2
    total_num = 0
    for batches in test_batches[60:]:
        for batch in batches:
            total_num += 1
            print(f'the {total_num} data')
            print(f'original track control is {batch[4:16]}')
            restored_with_generated_token, restored_with_target_token, restored_empty_fill = generate(model, batch, mask_bars, mask_tracks,mask_tensile,mask_diameter, device)


            restored_with_target_pm,_ = event_2midi(restored_with_target_token.tolist())
            restored_with_target_pm.write('restore_target.mid')


            restored_with_generated_pm,_ = event_2midi(restored_with_generated_token.tolist())
            restored_with_generated_pm.write('restore_generated.mid')

            restored_with_empty_pm,_ = event_2midi(restored_empty_fill.tolist())
            restored_with_empty_pm.write('restore_empty.mid')



            generated_tensile_category, generated_diameter_category,target_tensile_category,target_diameter_category = cal_bar_control('restore_target.mid', 'restore_generated.mid')

            for bar in mask_bars:
                print(f'bar {bar+1}, target tensile is {target_tensile_category[bar]} , generated tensile is {generated_tensile_category[bar]} \n'
                      f'bar {bar+1}, target diameter is {target_diameter_category[bar]} , generated diameter is {generated_diameter_category[bar]} \n')

            cal_track_control(restored_with_generated_token, restored_with_generated_pm)
            cal_track_control(restored_with_target_token, restored_with_target_pm)

            original_pm, _ = event_2midi(batch)
            original_pm.write('original.mid')

            cal_track_control(np.array(batch), original_pm)

            if diameter_control_change != 0:
                print('change diameter control')
                batch = change_control(batch, diameter_control_change, diameter_token,0,11,mask_bars=mask_bars)

                new_bar_control_restored_with_generated_token, new_bar_control_restored_with_target_token, new_bar_control_restored_empty_fill = generate(model, batch,
                                                                                                          mask_bars,
                                                                                                          mask_tracks,
                                                                                                    mask_tensile=[True],
                                                                                                    mask_diameter=[False],
                                                                                                    device=device)

                new_bar_control_restored_with_target_pm, _ = event_2midi(new_bar_control_restored_with_target_token.tolist())
                new_bar_control_restored_with_target_pm.write('new_bar_diameter_control_restore_target.mid')

                new_bar_control_restored_with_generated_pm, _ = event_2midi(new_bar_control_restored_with_generated_token.tolist())
                new_bar_control_restored_with_generated_pm.write('new_bar_diameter_control_restore_generated.mid')

                new_bar_control_restored_with_empty_pm, _ = event_2midi(new_bar_control_restored_empty_fill.tolist())
                new_bar_control_restored_with_empty_pm.write('new_bar_diameter_control_restore_empty.mid')

                generated_tensile_category, generated_diameter_category,target_tensile_category,target_diameter_category = cal_bar_control('new_bar_diameter_control_restore_target.mid', 'new_bar_diameter_control_restore_generated.mid')

                for bar in mask_bars:
                    print(
                        f'bar {bar+1}, target tensile is {target_tensile_category[bar] + diameter_control_change} , generated tensile is {generated_tensile_category[bar]} \n'
                        f'bar {bar+1}, target diameter is {target_diameter_category[bar]  + diameter_control_change} , generated diameter is {generated_diameter_category[bar]} \n')

            if tensile_control_change != 0:
                print('change tensile control')
                batch = change_control(batch, tensile_control_change, tensile_strain_token, 0, 11, mask_bars=mask_bars)

                new_bar_control_restored_with_generated_token, new_bar_control_restored_with_target_token, new_bar_control_restored_empty_fill = generate(
                    model, batch,
                    mask_bars,
                    mask_tracks,
                    mask_tensile=[False],
                    mask_diameter=[True],
                    device=device)

                new_bar_control_restored_with_target_pm, _ = event_2midi(
                    new_bar_control_restored_with_target_token.tolist())
                new_bar_control_restored_with_target_pm.write('new_bar_tensile_control_restore_target.mid')

                new_bar_control_restored_with_generated_pm, _ = event_2midi(
                    new_bar_control_restored_with_generated_token.tolist())
                new_bar_control_restored_with_generated_pm.write('new_bar_tensile_control_restore_generated.mid')

                new_bar_control_restored_with_empty_pm, _ = event_2midi(
                    new_bar_control_restored_empty_fill.tolist())
                new_bar_control_restored_with_empty_pm.write('new_bar_tensile_control_restore_empty.mid')

                generated_tensile_category, generated_tensile_category, target_tensile_category, target_tensile_category = cal_bar_control(
                    'new_bar_tensile_control_restore_target.mid', 'new_bar_tensile_control_restore_generated.mid')

                for bar in mask_bars:
                    print(
                        f'bar {bar+1}, target tensile is {target_tensile_category[bar] + tensile_control_change} , generated tensile is {generated_tensile_category[bar]} \n'
                        f'bar {bar+1}, target diameter is {target_diameter_category[bar] + tensile_control_change} , generated diameter is {generated_diameter_category[bar]} \n')

                # bar control change should not change track control
                # print('generated track token')
                # print(new_bar_control_restored_with_generated_token)
                cal_track_control(new_bar_control_restored_with_generated_token, new_bar_control_restored_with_generated_pm)
                print('target track token')
                cal_track_control(new_bar_control_restored_with_target_token, new_bar_control_restored_with_target_pm)

            # if track_control_change != 0:
            #     print('change track control')
            #     batch = change_control(batch, track_control_change, track_control_tokens,0,7)
            #
            #     new_track_control_restored_with_generated_token, new_track_control_restored_with_target_token, new_track_control_restored_empty_fill = generate(model, batch,
            #                                                                                               mask_bars,
            #                                                                                               mask_tracks,
            #                                                                                               device)
            #
            #     new_track_control_restored_with_target_pm, _ = event_2midi(new_track_control_restored_with_target_token.tolist())
            #     new_track_control_restored_with_target_pm.write('new_track_control_restore_target.mid')
            #
            #     new_track_control_restored_with_generated_pm, _ = event_2midi(new_track_control_restored_with_generated_token.tolist())
            #     new_track_control_restored_with_generated_pm.write('new_track_control_restore_generated.mid')
            #
            #     new_track_control_restored_with_empty_pm, _ = event_2midi(new_track_control_restored_empty_fill.tolist())
            #     new_track_control_restored_with_empty_pm.write('new_track_control_restore_empty.mid')
            #
            #     generated_tensile_category, generated_diameter_category,target_tensile_category,target_diameter_category = cal_bar_control('new_track_control_restore_target.mid', 'new_track_control_restore_generated.mid')
            #
            #     # track control change should not change bar control ?
            #     for bar in mask_bars:
            #         print(
            #             f'bar {bar}, target tensile is {target_tensile_category[bar]} , generated tensile is {generated_tensile_category[bar]} \n'
            #             f'bar {bar}, target diameter is {target_diameter_category[bar]} , generated diameter is {generated_diameter_category[bar]} \n')
            #
            #     # print('generated track token')
            #     # print(new_track_control_restored_with_generated_token)
            #
            #     cal_track_control(new_track_control_restored_with_generated_token, new_track_control_restored_with_generated_pm)
            #     print('target track token')
            #     # cal_track_control(new_track_control_restored_with_target_token, new_track_control_restored_with_target_pm)
            #     print(batch[4:16])

            # print(0)
    # test_dataset = ParallelLanguageDataset('', '',
    #                                        vocab, 0,
    #                                        0,
    #                                        2400,
    #                                        window_size,
    #                                        test_batches,
    #                                        test_batch_lengths,
    #                                        config['batch_size']['value'],
    #                                        .15,
    #                                        .3,
    #                                        .3,
    #                                        .3,
    #                                        .9,
    #                                        .3,
    #                                        3,
    #                                        0.5,
    #                                        span_ratio_separately_each_epoch,
    #                                        mask_bar_num_ratio=[0,0,0,0,
    #                                                            0,0,0,0,
    #                                                            0,0,0,0,
    #                                                            0,0,1,0],
    #                                        mask_track_num_ratio=[[.5, .25, .25], [.5, .5]],
    #                                        mask_bar_ctrl_token=False,
    #                                        pretraining=False,
    #                                        train_jointly=True,
    #                                        verbose=True)
    #
    # test_data_loader = DataLoader(test_dataset, batch_size=1,
    #                               collate_fn=lambda batch: collate_mlm(batch))
    #
    # ce_weight = torch.ones(vocab.vocab_size, device=device)
    # ce_weight[0] = 0
    # ce_weight[1] = 1
    # ce_weight[1] = 1
    # ce_weight[11:18] = 0
    # ce_weight[170:234] = 0
    # ce_loss = nn.CrossEntropyLoss(ignore_index=0, weight=ce_weight, reduction='none')
    #
    # ce_weight_all = torch.ones(vocab.vocab_size, device=device)
    # ce_weight_all[0] = 0
    #
    # tempo_weight = torch.zeros(vocab.vocab_size, device=device)
    # tempo_weight[11:18] = 1
    # tempo_loss = nn.CrossEntropyLoss(ignore_index=0, weight=tempo_weight, reduction='none')
    #
    # density_weight = torch.zeros(vocab.vocab_size, device=device)
    # density_weight[170:180] = 1
    # density_loss = nn.CrossEntropyLoss(ignore_index=0, weight=density_weight, reduction='none')
    #
    # occupation_weight = torch.zeros(vocab.vocab_size, device=device)
    # occupation_weight[180:190] = 1
    # occupation_loss = nn.CrossEntropyLoss(ignore_index=0, weight=occupation_weight, reduction='none')
    #
    # polyphony_weight = torch.zeros(vocab.vocab_size, device=device)
    # polyphony_weight[190:200] = 1
    # polyphony_loss = nn.CrossEntropyLoss(ignore_index=0, weight=polyphony_weight, reduction='none')
    #
    # pitch_register_weight = torch.zeros(vocab.vocab_size, device=device)
    # pitch_register_weight[200:210] = 1
    # pitch_register_loss = nn.CrossEntropyLoss(ignore_index=0, weight=pitch_register_weight, reduction='none')
    #
    # tensile_weight = torch.zeros(vocab.vocab_size, device=device)
    # tensile_weight[210:222] = 1
    # tensile_loss = nn.CrossEntropyLoss(ignore_index=0, weight=tensile_weight, reduction='none')
    #
    # diameter_weight = torch.zeros(vocab.vocab_size, device=device)
    # diameter_weight[222:234] = 1
    # diameter_loss = nn.CrossEntropyLoss(ignore_index=0, weight=diameter_weight, reduction='none')

    # for data in iter(test_data_loader):
    #
    #     minibatch_size = data['input'].size()[0]
    #
    #     for mini_batch_num in range(minibatch_size):
    #         with torch.no_grad():
    #             src, src_key_padding_mask = data['input'][mini_batch_num], None
    #
    #             src_masked_track = torch.sum(src == vocab.char2index('m_0')).item()
    #             tgt_inp = []
    #             tgt_out = data['target_out'][mini_batch_num]
    #             for _ in range(src_masked_track):
    #                 # i = 0
    #                 tgt_inp.append(vocab.char2index('m_0'))
    #                 while tgt_inp[-1] != vocab.char2index('<eos>'):
    #                     output = model_generate(model, src, tgt_inp).to(device)
    #                     # values, indices = torch.topk(output, 5)
    #                     # tgt_inp.append(int(indices[-1][0]))
    #                     index = sampling(output[-1], 0.9)
    #                     # print(index)
    #                     tgt_inp.append(index)
    #                     # i += 1
    #
    #
    #
    #
    #             generated_output = []
    #             target_output = []
    #
    #             # for i, token_idx in enumerate(tgt_inp):
    #             #     output_token = vocab.index2char(token_idx)
    #             #
    #             #     if i < tgt_out.size()[0]:
    #             #         target_idx = tgt_out[i].item()
    #             #     else:
    #             #         target_idx = vocab.char2index('<pad>')
    #             #
    #             #     target_token = vocab.index2char(target_idx)
    #
    #             generated_output = []
    #             target_output = []
    #             src_token = []
    #
    #             for i, token_idx in enumerate(tgt_inp):
    #                 output_token = vocab.index2char(token_idx)
    #                 generated_output.append(output_token)
    #
    #             for i, token_idx in enumerate(tgt_out):
    #                 target_token = vocab.index2char(token_idx.item())
    #                 target_output.append(target_token)
    #
    #             # print(f'generation {generated_output}')
    #             # print(f'target {target_output}')
    #
    #             for i, token_idx in enumerate(src):
    #                 src_token.append(vocab.index2char(token_idx.item()))
    #
    #             restored_with_generated_token, restored_with_target_token, restored_empty_fill = restore_marked_input(
    #                 src_token, generated_output, target_output)
    #
    #         break
    #     break


def model_generate(model, src, tgt):

    src = src.clone().detach().unsqueeze(0).long().to('cuda')
    tgt = torch.tensor(tgt).unsqueeze(0).to('cuda')
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    tgt_mask = tgt_mask.clone().detach().unsqueeze(0).to('cuda')


    output,weights = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                           tgt_mask=tgt_mask)

    return output.squeeze(0).to('cpu')


main()
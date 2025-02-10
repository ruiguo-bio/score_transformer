
from einops import rearrange

import pickle
import torch
import os

import argparse
import copy
import sys
import pretty_midi



import re
import yaml
import itertools

from vocab_colab import *


from model import ScoreTransformer

import tension_calculation



V0=120
V1=100
V2=60


def remi_2midi(events):
    track_num = [f'track_{num}' for num in range(3)]
    if events[1][0] == 't':
        # print(events)
        tempo_category = int(events[1][2])
        if tempo_category == len(tempo_bins) - 1:
            tempo = tempo_bins[tempo_category]
        else:
            tempo = (tempo_bins[tempo_category] + tempo_bins[tempo_category + 1]) / 2
    else:
        tempo = float(events[1])
    pm_new = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    numerator = int(events[0].split('/')[0])
    denominator = int(events[0].split('/')[1])
    time_signature = pretty_midi.TimeSignature(numerator, denominator, 0)
    pm_new.time_signature_changes = [time_signature]

    r = re.compile('i_\d')

    programs = list(filter(r.match, events))

    for track in programs:
        track = pretty_midi.Instrument(program=int(track.split('_')[-1]))
        pm_new.instruments.append(track)

    # add a fake note for duration dict calculation
    pm_new.instruments[0].notes.append(pretty_midi.Note(
        velocity=100, pitch=30, start=0, end=10))
    beats = pm_new.get_beats()
    pm_new.instruments[0].notes.pop()
    duration_name_to_time, duration_time_to_name, duration_times, bar_duration = get_note_duration_dict(
        beats[1] - beats[0], (time_signature.numerator, time_signature.denominator))
    sixteenth_duration = duration_name_to_time['sixteenth']
    curr_time = 0
    bar_num = 0
    bar_start_time = 0
    pitch_list = []
    for idx, event in enumerate(events):
        # print(idx,event)
        if event == 'bar':
            curr_time = bar_num * bar_duration
            bar_start_time = curr_time
            bar_num += 1
        if event in track_num:
            curr_time = bar_start_time
            current_track = event

        if event in step_token:
            curr_time = bar_start_time + int(event[2:]) * sixteenth_duration
        if event in pitch_tokens:
            pitch_list.append(int(event[2:]))

        if event in duration_single:
            end_time = curr_time + (int(event[2:])) * sixteenth_duration
            start_time = curr_time
            for pitch in pitch_list:
                if current_track == 'track_0':
                    vel = V0
                elif current_track == 'track_1':
                    vel = V1
                else:
                    vel = V2
                note = pretty_midi.Note(velocity=vel, pitch=pitch,
                                        start=start_time, end=end_time)
                pm_new.instruments[int(current_track[6])].notes.append(note)
            pitch_list = []

    return pm_new





def cal_tension(pm,key_names=None):


    result = tension_calculation.extract_notes(pm, 3)

    if result:

        pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result
    else:
        return None

    if key_names:
        key_name = [key_names]
    else:
        key_name = tension_calculation.all_key_names



    result = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices, -1, key_name,sixteenth_time,pm)

    if result:
        tensiles, diameters, key_name,\
        changed_key_name, key_change_beat = result
    else:
        return None

    tensile_category = to_category(tensiles,tensile_bins)
    diameter_category = to_category(diameters, diameter_bins)

    # print(f'key is {key_name}')

    return tensile_category, diameter_category,key_name

def note_density(track_events, track_length):
    densities = []
    tracks = track_events.keys()
    # print(tracks)
    for track_name in tracks:
        # print(track_name)
        note_num = 0
        this_track_events = track_events[track_name]
        # print(this_track_events)
        for track_event in this_track_events:
            for event_index in range(len(track_event) - 1):
                if track_event[event_index][0] == 'p' and track_event[event_index + 1][0] != 'p':
                    note_num += 1
        #         print(note_num / track_length)
        densities.append(note_num / track_length)
    return densities


def occupation_polyphony_rate(pm):
    occupation_rate = []
    polyphony_rate = []
    beats = pm.get_beats()
    fs = 4 / (beats[1] - beats[0])

    for instrument in pm.instruments:
        piano_roll = instrument.get_piano_roll(fs=fs)
        if piano_roll.shape[1] == 0:
            occupation_rate.append(0)
        else:
            occupation_rate.append(np.count_nonzero(np.any(piano_roll, 0)) / piano_roll.shape[1])
        if np.count_nonzero(np.any(piano_roll, 0)) == 0:
            polyphony_rate.append(0)
        else:
            polyphony_rate.append(
                np.count_nonzero(np.count_nonzero(piano_roll, 0) > 1) / np.count_nonzero(np.any(piano_roll, 0)))

    return occupation_rate, polyphony_rate


def to_category(array, bins):
    result = []
    for item in array:
        result.append(int(np.where((item - bins) >= 0)[0][-1]))
    return result


def pitch_register(track_events):
    registers = []
    tracks = track_events.keys()
    # print(tracks)
    for track_name in tracks:
        # print(track_name)
        register = []
        this_track_events = track_events[track_name]
        # print(this_track_events)
        for track_event in this_track_events:
            for event in track_event:
                if event[0] == 'p':
                    register.append(int(event[2:]))
        #         print(note_num / track_length)
        # print(np.mean(register))
        if len(register) == 0:
            registers.append(0)
        else:
            registers.append(int((np.mean(register) - 21) / 11))
    return registers


def cal_track_control(file_events,pm):


    r = re.compile('i_\d')

    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)



    file_events = np.array(file_events)

    #     print(f'number of bars is {len(bar_pos)}')
    #     print(f'time signature is {file_event[1]}')
    bar_length = int(file_events[0][0])
    bar_pos = np.where(file_events == 'bar')[0]
    if bar_length != 6:
        bar_length = bar_length * 4 * len(bar_pos)
    else:
        bar_length = bar_length / 2 * 4 * len(bar_pos)
    #     print(f'bar length is {bar_length}')

    track_events = {}

    for i in range(min(num_of_tracks,3)):
        track_events[f'track_{i}'] = []
    track_names = list(track_events.keys())
    if len(track_names) == 0:
        return None

    for bar_index in range(len(bar_pos) - 1):
        bar = bar_pos[bar_index]
        next_bar = bar_pos[bar_index + 1]
        bar_events = file_events[bar:next_bar]
        #         print(bar_events)

        track_pos = []

        for track_name in track_names:
            if len(np.where(track_name == bar_events)[0]) == 0:
                print(bar_events)
                print(f'track name is {track_name}')
                return None
            else:
                track_pos.append(np.where(track_name == bar_events)[0][0])
        #         print(track_pos)
        track_index = 0
        for track_index in range(len(track_names) - 1):
            track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
            track_events[track_names[track_index]].append(track_event)
        #             print(track_event)
        else:
            if len(track_names) != 1:
                track_index += 1
            else:
                track_index = 0

            track_event = bar_events[track_pos[track_index]:]
            #             print(track_event)
            track_events[track_names[track_index]].append(track_event)

    densities = note_density(track_events, bar_length)
    density_category = to_category(densities, control_bins)

    occupation_rate, polyphony_rate = occupation_polyphony_rate(pm)
    occupation_category = to_category(occupation_rate, control_bins)
    polyphony_category = to_category(polyphony_rate, control_bins)
    pitch_register_category = pitch_register(track_events)

    density_token = [f'd_{category}' for category in density_category]
    occupation_token = [f'o_{category}' for category in occupation_category]
    polyphony_token = [f'y_{category}' for category in polyphony_category]
    pitch_register_token = [f'r_{category}' for category in pitch_register_category]

    track_control_tokens = density_token + occupation_token + polyphony_token + pitch_register_token


    return track_control_tokens


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

def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word



def sampling_step_single(logit,vocab, p=None, t=1.0,no_pitch=False,no_duration=False,no_step=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:

        logit = np.array([-100 if i in vocab.pitch_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in vocab.duration_only_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_step:
        logit = np.array([-100 if i in vocab.step_indices else logit[i] for i in range(vocab.vocab_size)])


    logit = np.array([-100 if i in vocab.program_indices + vocab.structure_indices + vocab.time_signature_indices + vocab.tempo_indices  else logit[i] for i in range(vocab.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


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

# mask_tracks:0,1,2
# mask_bars:[4,5,6,7]
def mask_bar_and_track(event,vocab,mode,mask_tracks=0,mask_bars=[8]):

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


    if mask_mode == 0:

        if len(bar_poses) < mask_bars[-1]:
            return None


        for mask_bar in mask_bars:

            track_mask_poses = mask_tracks

            for track_mask_pos in track_mask_poses:
                mask_track_names.append(track_mask_pos)
                mask_bar_names.append(mask_bar)
                bar_with_track_poses[mask_bar][track_mask_pos]
                masked_indices_pairs.append(bar_with_track_poses[mask_bar][track_mask_pos])
    elif mask_mode == 1:
        # mask whole tracks
        # if track_nums == 1:
        #     return None

        # track_mask_number = np.random.randint(0, track_nums-1)
        if track_nums > mask_tracks[-1]:
            track_mask_poses = mask_tracks
        else:
            return None
        for bar_num, tracks_in_a_bar in enumerate(bar_with_track_poses):
            for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                if track_pos in track_mask_poses:
                    mask_bar_names.append(bar_num)
                    mask_track_names.append(track_pos)
                    masked_indices_pairs.append(track_star_end_poses)

    else:
        # mask whole bars

        bar_mask_poses = mask_bars

        for bar_mask_pos in bar_mask_poses:
            for track_name in range(track_nums):
                mask_bar_names.append(bar_mask_pos)
                mask_track_names.append(track_name)
                masked_indices_pairs.append(bar_with_track_poses[bar_mask_pos][track_name])

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





def get_note_duration_dict(beat_duration,curr_time_signature):
    duration_name_to_time = {}
    if curr_time_signature[1] == 4:
        # 4/4, 2/4, 3/4
        quarter_note_duration = beat_duration
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        if curr_time_signature[0] >= 4:
            whole_note_duration = 4 * quarter_note_duration
        bar_duration = curr_time_signature[0] * quarter_note_duration

    else:
        # 6/8

        quarter_note_duration = beat_duration / 3 * 2
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4

        bar_duration = curr_time_signature[0] * eighth_note_duration

    duration_name_to_time['half'] = half_note_duration
    duration_name_to_time['quarter'] = quarter_note_duration
    duration_name_to_time['eighth'] = eighth_note_duration
    duration_name_to_time['sixteenth'] = sixteenth_note_duration

    basic_names = duration_name_to_time.keys()
    name_pairs = itertools.combinations(basic_names, 2)
    name_triple = itertools.combinations(basic_names, 3)
    name_quadruple = itertools.combinations(basic_names, 4)

    for name1,name2 in name_pairs:
        duration_name_to_time[name1+'_'+name2] = duration_name_to_time[name1] + duration_name_to_time[name2]

    for name1, name2,name3 in name_triple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3] = duration_name_to_time[name1] + duration_name_to_time[name2] + duration_name_to_time[name3]

    for name1, name2, name3, name4 in name_quadruple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3 + '_' + name4] = duration_name_to_time[name1] + duration_name_to_time[
            name2] + duration_name_to_time[name3] + duration_name_to_time[name4]


    duration_name_to_time['zero'] = 0


    if curr_time_signature[0] >= 4 and curr_time_signature[1] == 4:
        duration_name_to_time['whole'] = whole_note_duration

    duration_time_to_name = {v: k for k, v in duration_name_to_time.items()}

    duration_times = np.sort(np.array(list(duration_time_to_name.keys())))
    return duration_name_to_time,duration_time_to_name,duration_times,bar_duration


def restore_marked_input(src_token, generated_output):
    src_token = np.array(src_token, dtype='<U9')

    # restore with generated output
    restored_with_generated_token = src_token.copy()

    generated_output = np.array(generated_output)

    generation_mask_indices = np.where(generated_output == 'm_0')[0]

    if len(generation_mask_indices) == 1:

        mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
        generated_result_sec = generated_output[generation_mask_indices[0] + 1:]

        #         print(len(generated_result_sec))
        restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
        for token in generated_result_sec[::-1]:
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

            for token in generated_result_sec[::-1]:
                #                 print(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)

        else:
            #         print(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i + 1] + 1:]

            #             print(len(generated_result_sec))
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
            for token in generated_result_sec[::-1]:
                #                 print(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)

    return restored_with_generated_token


def generation_all(model, events, device, vocab, mask_mode,mask_tracks,mask_bars):
    result = mask_bar_and_track(events, vocab, mask_mode,mask_tracks,mask_bars)
    if result is None:
        return result
    src, tgt_out, mask_track_names, mask_bar_names = result


    src_masked_nums = np.sum(src == vocab.char2index('m_0'))
    tgt_inp = []
    total_generated_events = []

    if src_masked_nums == 0:
        return None
    total_corrected_times = 0
    corrected_times = 0
    with torch.no_grad():
        mask_idx = 0
        while mask_idx < src_masked_nums:

            this_tgt_inp = []
            is_time_correct = False
            this_tgt_inp.append(vocab.char2index('m_0'))
            this_generated_events = []
            this_generated_events.append('m_0')
            total_grammar_correct_times = 0

            no_pitch = True
            no_step = False
            no_duration = True


            while this_tgt_inp[-1] != vocab.char2index('<eos>') and len(this_tgt_inp) < 100:


                if len(this_tgt_inp) == 1:
                    mask_track_name = 'track_' + f'{str(mask_track_names[mask_idx])}'
                    track_idx = vocab.char2index(mask_track_name)

                    this_tgt_inp.append(track_idx)
                    this_generated_events.append(mask_track_name)


                    continue

                output, weight = model_generate(model, torch.tensor(src), tgt_inp + this_tgt_inp, device,
                                                return_weights=True)
                


               
                if no_pitch and no_duration:
                    index = sampling_step_single(output[-1], vocab, no_pitch=no_pitch,no_step=no_step,no_duration=no_duration)
                    sampling_times = 0
                    # # step
                    while index not in vocab.step_indices and index != vocab.eos_index:
                        index = sampling_step_single(output[-1], vocab,  no_pitch=no_pitch,no_step=no_step,no_duration=no_duration)
                        sampling_times += 1
                        total_grammar_correct_times += 1
                        if sampling_times > 10:
                            print('empty track here')
                            break

                    event = vocab.index2char(index)


                    no_pitch = False
                    no_duration = True
                    no_step = True

                # pitch
                elif no_step and no_duration:

                    index = sampling_step_single(output[-1], vocab, no_step=no_step,
                                                 no_duration=no_duration)
                    sampling_times = 0
                    while index not in vocab.pitch_indices:
                        index = sampling_step_single(output[-1], vocab, no_step=no_step, no_duration=no_duration)
                        sampling_times += 1
                        total_grammar_correct_times += 1
                        if sampling_times > 10:
                            print('pitch failed here')
                            break
                    event = vocab.index2char(index)

                    no_duration = False
                    no_step = True
             
                elif no_step:

                    index = sampling_step_single(output[-1], vocab, no_step=no_step)
                    sampling_times = 0
                    while index in vocab.step_indices:
                        index = sampling_step_single(output[-1], vocab,  no_step=no_step)
                        sampling_times += 1
                        total_grammar_correct_times += 1
                        if sampling_times > 10:
                            print('step failed here')
                            break
                    event = vocab.index2char(index)
                    if index in vocab.duration_only_indices:

                        no_pitch = True
                        no_duration = True
                        no_step = False
                else:
                    pass


                this_tgt_inp.append(index)
                this_generated_events.append(event)

            
            mask_idx += 1
            tgt_inp.extend(this_tgt_inp[:-1])
            total_generated_events.extend(this_generated_events[:-1])

    src_token = []
    
    for i, token_idx in enumerate(src):
        src_token.append(vocab.index2char(token_idx.item()))

    tgt_output_events = []
    for i, token_idx in enumerate(tgt_out):
        if token_idx in vocab.structure_indices[1:]:
            tgt_output_events.append('m_0')
        if token_idx != vocab.char2index('<eos>'):
            tgt_output_events.append(vocab.index2char(token_idx.item()))

    return restore_marked_input(src_token, total_generated_events),restore_marked_input(src_token, tgt_output_events), mask_track_names, mask_bar_names

def get_args(default='.'):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--platform', default='local', type=str,
                        help="local mode")
    parser.add_argument('-v', '--vocab_mode', default=0, type=int,
                        help="vocab mode")

    parser.add_argument('-c', '--cuda', default=0, type=int,
                        help="cuda")

    parser.add_argument('-m', '--track_mode', default=False, type=bool,
                        help="track_mode")

    parser.add_argument('-t', '--mask_tracks', default=0, type=int,
                        help="mask_tracks")

    # parser.add_argument('-b', '--mask_bars', default="7,8", type=str,
    #                     help="mask_bars")

    parser.add_argument('-l', '--control_number', default=0, type=int,
                        help="control number")

    parser.add_argument('-b', '--bar_mode', default=False, type=bool,
                        help="bar mode")




    return parser.parse_args()


#
# pm = pretty_midi.PrettyMIDI('/home/data/guorui/score_transformer/evaluation_all/rest_multi_control_2_reverse/original_e070b3b932d24866992b8ee30aca7ff1_event_idx_9.mid')
# tensile_category, diameter_category, tension_category,key_name = cal_tension(pm)
#


control_list = ['key', 'tensile','diameter','density',
                'polyphony','occupation']

vocab = WordVocab(1,control_list)

selected_number = 100
output_folder_prefix = './output'
output_folder = f'{output_folder_prefix}/'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder + '/original', exist_ok=True)
os.makedirs(output_folder + '/change', exist_ok=True)


batches = './step_single_control_test_batches'
batch_names = './step_single_control_test_batch_names'



config_folder = './config/'

with open(os.path.join(config_folder, "files/config.yaml")) as file:

    config = yaml.full_load(file)

model = ScoreTransformer(vocab.vocab_size, config['d_model']['value'], config['nhead']['value'],
                         config['num_encoder_layers']['value'],
                         config['num_encoder_layers']['value'], 2048, 2400,
                         0.1, 0.1)

checkpoint = os.path.join(config_folder, f"files/checkpoint_9")
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

model_dict = torch.load(checkpoint, map_location=device)

model_state = model_dict['model_state_dict']
# optimizer_state = model_dict['optimizer_state_dict']

from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in model_state.items():
#     name = k[7:]  # remove `module.`
#     new_state_dict[name] = v

# new_state_dict = model_state

model.to(device)

model.load_state_dict(model_state)

# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)

window_size = int(16 / 2)

#

batches = pickle.load(open(batches, 'rb'))


batches_name = pickle.load(open(batch_names, 'rb'))

nums = 0

selected_indices = np.random.choice(len(batches), selected_number, replace=False)






for song_idx in selected_indices:

    mask_tracks = []
    new_track_control = ''
    selected_key_name = ''
    mask_bars = []


    failed_times = 0
    regenerate = False
    generate_times = 0

    key_list = []
    one_batches = batches[song_idx]

    for one_batch in one_batches:
        remove_idx = []
        for idx, token in enumerate(one_batch):
            if token not in vocab.control_tokens and \
                    token not in vocab.basic_tokens:
                remove_idx.append(idx)

        for idx in remove_idx[::-1]:
            one_batch.pop(idx)


    selected_track = -1
    # one_batches = batches[669]
    nums += 1


    # succeed = False
    song_name = batches_name[song_idx]
    print(f'working on {nums}th song {song_name}')
    total_num = len(one_batches)
    if total_num > 30:
        print(f'skip {batches_name[song_idx]} with length {total_num}')
        continue
    # while not succeed and generate_times < 10:
    #     generate_times += 1
    #     print(f'generate times is {generate_times}')

    valid = True



    for control_phase in control_phases:
        if control_phase == 'original':
            print(f'no change control test')
        else:
            print(f'change control test')

        total_generated_tokens = []
        total_original_tokens = []

        idx_number = 0
        while idx_number < total_num:
            print(f'the {idx_number}th number')
            if not regenerate:
                batch = copy.copy(one_batches[idx_number])
            # song_name = batch_names[idx].split('_')[0]

            # print(f'{song_name}  num {i}')
            # batch = one_batch
                if not isinstance(batch,list):
                    batch = batch.tolist()

                r = re.compile('i_\d')


                track_program = list(filter(r.match, batch))
                track_nums = len(track_program)





            if control_number in [2,3]:
                if control_phase == 'original':
                    if bar_mode:
                        bar_poses = np.where(np.array(batch) == 'bar')[0]
                        mask_bars = [np.random.choice(len(bar_poses))]
                        mask_mode = 2
                    else:
                        if len(mask_tracks) == 0:
                            mask_tracks = [np.random.randint(0, track_nums)]
                            mask_mode = 1
                else:
                    bar_poses = np.where(np.array(batch) == 'bar')[0]


                    if not regenerate:
                        bar_mask_number = np.random.choice(int(len(bar_poses)/2))
                        mask_bars = np.sort(np.random.choice(len(bar_poses), size=bar_mask_number + 1, replace=False))
                        mask_mode = 2

            if control_number in [4,6]:
                if control_phase == 'original':
                    if bar_mode:
                        bar_poses = np.where(np.array(batch) == 'bar')[0]
                        mask_bars = [np.random.choice(len(bar_poses))]
                        mask_mode = 2
                    else:
                        if len(mask_tracks) == 0:
                            mask_tracks = [np.random.randint(0, track_nums)]
                            mask_mode = 1
                else:
                    if len(mask_tracks) == 0:
                        mask_tracks = [np.random.randint(0, track_nums)]
                    track_mode = mask_mode = 1

            if control_number == 5:
                if control_phase == 'control':
                    if track_nums < 3:
                        idx_number += 1
                        continue
                    mask_tracks = [2]
                    track_mode = mask_mode = 1
                else:
                    if bar_mode:
                        bar_poses = np.where(np.array(batch) == 'bar')[0]
                        mask_bars = [np.random.choice(len(bar_poses))]
                        mask_mode = 2
                    else:
                        if len(mask_tracks) == 0:
                            mask_tracks = [np.random.randint(0, track_nums)]
                            mask_mode = 1

            if control_number in [7,8,9,10,11]:
                if len(mask_tracks) == 0:
                    mask_tracks = [np.random.randint(0, track_nums)]
                mask_mode = 1

            # if track_mode:
            #     if track_nums < mask_track:
            #         print(f'not enough track, continue')
            # else:
            #     bar_nums = len(np.where(np.array(batch) == 'bar')[0])
            #     if bar_nums <= mask_bars[-1]+1:
            #         print(f'not enough bar, continue')
            # batch[1] = 't_1'
            # result = generation(model, batch, device, vocab, 1)
            for event in batch[3:]:
                if event not in vocab.char_lst:
                    print(f'not in vocab event {event}')
                    valid = False
                    break

                # if event == 'continue':
                #     print(event)

            if not valid:
                idx_number += 1
                failed_times = 0
                continue

            # if int(batch[0][2]) != 8:
            #     continue

            # continue

            if control_phase == 'change':
                if control_number == 1:

                    print(f'key control change')
                    if selected_key_name == '':
                        selected_key_name = np.random.choice(all_key_names)

                    track_num_pos = np.where(track_program[0] == np.array(batch))[0][0]

                    # change token
                    original_key_token = batch[2]
                    original_key_name = token_to_key[original_key_token]
                    # original_key_name = original_key_name.replace(' ', '_')
                    batch[2] = key_to_token[selected_key_name]

                    print(f'change key from {token_to_key[original_key_token]} to {selected_key_name}')

                if control_number in [2,3]:
                    original_tensions = []
                    changed_tensions = []
                    tension_original_changed_diffs = []
                    tension_generated_changed_diffs = []

                    other_original_tensions = []

                    other_tension_original_diffs = []


                    if control_number == 2:
                        tension_pos_diff = 1
                    if control_number == 3:
                        tension_pos_diff = 2


                    if control_number == 2:

                        # tension_pos_diff = 1
                        changed_control_name = 'tensile'

                    if control_number == 3:

                        # tension_pos_diff = 2
                        changed_control_name = 'diameter'


                    for mask_bar_num in mask_bars:


                        if not regenerate:
                            if control_number == 2:
                                original_tension_token = batch[bar_poses[mask_bar_num] + 1]
                                original_level = int(original_tension_token.split('_')[-1])
                                # original_other_token = batch[bar_poses[mask_bar_num] + 2]

                            if control_number == 3:
                                original_tension_token = batch[bar_poses[mask_bar_num] + 2]
                                original_level = int(original_tension_token.split('_')[-1])
                                # original_other_token = batch[bar_poses[mask_bar_num] + 1]

                            # if control_number == 2:
                            #     if original_level < 4:
                            #         new_bar_control = original_tension_token.split('_')[0] + '_' + str(original_level + 3)
                            #     elif original_level > 3:
                            #         new_bar_control = original_tension_token.split('_')[0] + '_' + str(original_level - 3)
                            #     else:
                            #         new_bar_control = original_tension_token
                            # if control_number == 3:
                            #     if original_level < 4:
                            #         new_bar_control = original_tension_token.split('_')[0] + '_' + str(original_level + 3)
                            #     elif original_level > 5:
                            #         new_bar_control = original_tension_token.split('_')[0] + '_' + str(original_level - 3)
                            #     else:
                            #         new_bar_control = original_tension_token

                            new_bar_control = np.random.choice(vocab.name_to_tokens[changed_control_name])
                            while abs(int(int(original_tension_token.split('_')[-1])) - int(new_bar_control.split('_')[-1])) > 4:
                                new_bar_control = np.random.choice(vocab.name_to_tokens[changed_control_name])
                            # while new_bar_control != original_tension_token and int(new_bar_control.split('_')[-1]) > 8:
                            #     new_bar_control = np.random.choice(vocab.name_to_tokens[changed_control_name])

                            original_tensions.append(original_tension_token)
                            changed_tensions.append(new_bar_control)
                            # other_original_tensions.append(original_other_token)
                            if control_number == 2:
                                batch[bar_poses[mask_bar_num] + 1] = new_bar_control
                            if control_number == 3:
                                batch[bar_poses[mask_bar_num] + 2] = new_bar_control

                            print(
                                f'change bar {mask_bar_num} {changed_control_name} from {original_tension_token} to {new_bar_control}')
                        else:

                            original_tension_token = batch[bar_poses[mask_bar_num] + tension_pos_diff]
                            new_bar_control = original_tension_token

                            original_tensions.append(original_tension_token)
                            changed_tensions.append(new_bar_control)
                            print(
                                f'change bar {mask_bar_num} {changed_control_name} from {original_tension_token} to {new_bar_control}')



                if control_number in [4,5,6]:

                    track_control_end_pos = np.where(np.array(batch) == track_program[0])[0][0]
                    original_track_control = batch[3:track_control_end_pos]

                    if control_number == 4:
                        selected_control_name = 'density'
                    if control_number == 5:
                        selected_control_name = 'polyphony'
                    if control_number == 6:
                        selected_control_name = 'occupation'

                    track_num_pos = np.where(track_program[0] == np.array(batch))[0][0]
                    selected_track = mask_tracks[0]
                    for j, token in enumerate(original_track_control):
                        if vocab.token_class_ranges[vocab.char2index(token)] == selected_control_name and \
                                j % track_nums == selected_track:

                            # change token
                            original_track_token = original_track_control[j]

                            if new_track_control == '':
                                new_track_control = np.random.choice(vocab.name_to_tokens[selected_control_name])
                                # while abs(int(new_track_control[-1]) - int(original_track_token[-1])) < 3:
                                # new_track_control = np.random.choice(vocab.name_to_tokens[selected_control_name])

                            batch[3 + j] = new_track_control
                            print(
                                f'change track {selected_track} control from {original_track_token} to {new_track_control}')
                            break

                    original_control_diff = int(original_track_token[-1]) - int(new_track_control[-1])

                    original_track_control[j] = new_track_control



            result = generation_all(model, batch, device, vocab, mask_mode,mask_tracks,mask_bars)

            if result is None:
                print('failed')
                failed_times += 1
                if failed_times > 10:
                    print(f'failed number is {failed_times}')
                    idx_number += 1
                    failed_times = 0
                continue
    # def compare_result(result):
            if regenerate:
                restored_with_generated_token,_, mask_track_names, mask_bar_names = result

            else:
                restored_with_generated_token, restored_with_target_token, mask_track_names, mask_bar_names = result
                restored_with_target_token = restored_with_target_token.tolist()

            original_bar_pos = np.where(np.array(restored_with_target_token) == 'bar')[0]
            restored_with_generated_token = restored_with_generated_token.tolist()
            mask_track_names = list(set(mask_track_names))
            generated_bar_pos = np.where(np.array(restored_with_generated_token) == 'bar')[0]


            # if track_mode and len(restored_with_generated_token) < 30:
            #     print('total generated token too short')
            #     continue
            if not bar_mode:
                if control_phase == 'original' or (control_phase == 'change' and control_number not in [2,3]):
                    if len(generated_bar_pos) > 8:

                        if idx_number == 0:
                            total_generated_tokens.extend(restored_with_generated_token)
                        else:
                            total_generated_tokens.extend(restored_with_generated_token[generated_bar_pos[8]:])

                    else:
                        if idx_number == 0:
                            total_generated_tokens.extend(restored_with_generated_token)
                        # else:
                        #     total_generated_tokens.extend(restored_with_generated_token)

                    if len(original_bar_pos) > 8:

                        if idx_number == 0:
                            total_original_tokens.extend(restored_with_target_token)
                        else:
                            total_original_tokens.extend(restored_with_target_token[original_bar_pos[8]:])

                    else:
                        if idx_number == 0:
                            total_original_tokens.extend(restored_with_target_token)
                        else:
                            total_original_tokens.extend(restored_with_target_token)



            generated_pm = remi_2midi(restored_with_generated_token)
            original_pm = remi_2midi(restored_with_target_token)


            # if not track_mode:
            #     generated_pm.write(f'{output_folder}/generated_{song_name}_number_{i}.mid')
            #
            #     original_pm.write(f'{output_folder}/original_{song_name}_number_{i}.mid')

            if len(generated_pm.get_beats()) < 6 or len(original_pm.get_beats()) < 6:
                print('too short')
                failed_times += 1
                if failed_times > 10:
                    print(f'failed number is {failed_times}')
                    idx_number += 1
                    failed_times = 0
                continue


            #
            # for i in range(0, len(generated_track_control), track_nums):
            #     for track_name in mask_track_names:
            #         if original_track_control[i + track_name][0] == 'o':
            #             if abs(int(original_track_control[i + track_name][-1]) - int(generated_track_control[i + track_name][-1])) < 3:
            #                 succeed = True
            #                 break
            #     if succeed:
            #         break
            # if succeed:


            if control_phase == 'original':
                if bar_mode:
                    print(f'mask bar is {mask_bars}')
                    generated_feature = cal_features('', bar=mask_bars, pm=generated_pm)
                    original_feature = cal_features('', bar=mask_bars, pm=original_pm)
                else:

                    print(f'mask track is {mask_track_names}')
                    generated_feature = cal_features('', track=mask_tracks[0],pm=generated_pm)
                    original_feature = cal_features('', track=mask_tracks[0],pm=original_pm)

                if generated_feature is None or original_feature is None:
                    idx_number += 1
                    failed_times = 0
                    continue

                idx_number += 1
                failed_times = 0


                
                if bar_mode:
                    used_pitch_number, used_notes_number, pitch_range, \
                    chromagram, pitch_intecrervals_hist, duration_hist, \
                    onset_interval_hist = generated_feature
                    
                    bar_generated_dict['used_pitch_number'].append(used_pitch_number)
                    bar_generated_dict['used_notes_number'].append(used_notes_number)
                    bar_generated_dict['pitch_range'].append(pitch_range)
                    bar_generated_dict['chromagram'].append(chromagram)
                    bar_generated_dict['pitch_intervals_hist'].append(pitch_intervals_hist)
                    bar_generated_dict['duration_hist'].append(duration_hist)
                    bar_generated_dict['onset_interval_hist'].append(onset_interval_hist)

                    used_pitch_number, used_notes_number, pitch_range, \
                    chromagram, pitch_intervals_hist, duration_hist, \
                    onset_interval_hist = original_feature

                    bar_original_dict['used_pitch_number'].append(used_pitch_number)
                    bar_original_dict['used_notes_number'].append(used_notes_number)
                    bar_original_dict['pitch_range'].append(pitch_range)
                    bar_original_dict['chromagram'].append(chromagram)
                    bar_original_dict['pitch_intervals_hist'].append(pitch_intervals_hist)
                    bar_original_dict['duration_hist'].append(duration_hist)
                    bar_original_dict['onset_interval_hist'].append(onset_interval_hist)
                else:
                    used_pitch_number, used_notes_number, pitch_range, \
                    chromagram, pitch_intervals_hist, duration_hist, \
                    onset_interval_hist = generated_feature


                    generated_dict[mask_tracks[0]]['used_pitch_number'].append(used_pitch_number)
                    generated_dict[mask_tracks[0]]['used_notes_number'].append(used_notes_number)
                    generated_dict[mask_tracks[0]]['pitch_range'].append(pitch_range)
                    generated_dict[mask_tracks[0]]['chromagram'].append(chromagram)
                    generated_dict[mask_tracks[0]]['pitch_intervals_hist'].append(pitch_intervals_hist)
                    generated_dict[mask_tracks[0]]['duration_hist'].append(duration_hist)
                    generated_dict[mask_tracks[0]]['onset_interval_hist'].append(onset_interval_hist)

                    used_pitch_number, used_notes_number, pitch_range, \
                    chromagram, pitch_intervals_hist, duration_hist, \
                    onset_interval_hist = original_feature

                    original_dict[mask_tracks[0]]['used_pitch_number'].append(used_pitch_number)
                    original_dict[mask_tracks[0]]['used_notes_number'].append(used_notes_number)
                    original_dict[mask_tracks[0]]['pitch_range'].append(pitch_range)
                    original_dict[mask_tracks[0]]['chromagram'].append(chromagram)
                    original_dict[mask_tracks[0]]['pitch_intervals_hist'].append(pitch_intervals_hist)
                    original_dict[mask_tracks[0]]['duration_hist'].append(duration_hist)
                    original_dict[mask_tracks[0]]['onset_interval_hist'].append(onset_interval_hist)

                if not bar_mode:
                    r = re.compile('i_\d')

                    generated_track_control = cal_track_control(restored_with_generated_token,
                                                                generated_pm)

                    original_track_control = cal_track_control(restored_with_target_token,
                                                               original_pm)

                    track_program = list(filter(r.match, batch))
                    track_nums = len(track_program)

                    for i in range(0, len(generated_track_control), track_nums):
                        for track_name in mask_track_names:
                            # print((original_track_control[i + track_name], generated_track_control[i + track_name]))
                            if original_track_control[i + track_name][0] == 'd':

                                no_change_track_diff_dict[mask_tracks[0]]['density'].append(
                                    int(original_track_control[i + track_name][-1]) - int(generated_track_control[i + track_name][-1]))

                                no_change_generated_track_dict[mask_tracks[0]]['density'].append(int(
                                        generated_track_control[i + track_name][-1]))

                                no_change_original_track_dict[mask_tracks[0]]['density'].append(
                                    int(original_track_control[i + track_name][-1]))
                            if original_track_control[i + track_name][0] == 'o':

                                no_change_track_diff_dict[mask_tracks[0]]['occupation'].append(
                                    int(original_track_control[i + track_name][-1]) - int(
                                        generated_track_control[i + track_name][-1]))

                                no_change_generated_track_dict[mask_tracks[0]]['occupation'].append(int(
                                    generated_track_control[i + track_name][-1]))

                                no_change_original_track_dict[mask_tracks[0]]['occupation'].append(
                                    int(original_track_control[i + track_name][-1]))
                            if original_track_control[i + track_name][0] == 'y':

                                no_change_track_diff_dict[mask_tracks[0]]['polyphony'].append(
                                    int(original_track_control[i + track_name][-1]) - int(
                                        generated_track_control[i + track_name][-1]))

                                no_change_generated_track_dict[mask_tracks[0]]['polyphony'].append(int(
                                    generated_track_control[i + track_name][-1]))

                                no_change_original_track_dict[mask_tracks[0]]['polyphony'].append(
                                    int(original_track_control[i + track_name][-1]))
                            if original_track_control[i + track_name][0] == 'r':

                                no_change_track_diff_dict[mask_tracks[0]]['pitch'].append(
                                    int(original_track_control[i + track_name][-1]) - int(
                                        generated_track_control[i + track_name][-1]))

                                no_change_generated_track_dict[mask_tracks[0]]['pitch'].append(int(
                                    generated_track_control[i + track_name][-1]))

                                no_change_original_track_dict[mask_tracks[0]]['pitch'].append(
                                    int(original_track_control[i + track_name][-1]))
            else:
                if control_number == 1:
                    result = cal_tension(generated_pm)
                    if result:
                        _, _, generated_key = result
                        idx_number += 1
                        failed_times = 0
                    else:
                        failed_times += 1
                        if failed_times > 10:
                            print(f'failed number is {failed_times}')
                            idx_number += 1
                            failed_times = 0
                        continue
                    print(f'generated key is {generated_key}')
                    key_list.append((original_key_name,selected_key_name,generated_key))



                if control_number in [2,3]:
                    result = cal_tension(generated_pm)
                    if control_number == 2:

                        if result is not None:
                            tensions,tensions_other,_ = result

                        else:
                            failed_times += 1
                            if failed_times > 10:
                                print(f'failed number is {failed_times}')
                                idx_number += 1
                                failed_times = 0
                            continue
                    else:
                        if result is not None:
                            tensions_other,tensions, _, _ = result

                        else:
                            failed_times += 1
                            if failed_times > 10:
                                print(f'failed number is {failed_times}')
                                idx_number += 1
                                failed_times = 0
                            continue


                    if len(tensions) <= mask_bars[-1]:
                        failed_times += 1
                        if failed_times > 10:
                            print(f'failed number is {failed_times}')
                            idx_number += 1
                            failed_times = 0
                        continue
                    else:
                        new_mask_bars = []
                        for idx, mask_bar in enumerate(mask_bars):
                            if generate_times == 0:
                                tension_diff_original_changed = abs(int(original_tensions[idx].split('_')[-1]) - int(changed_tensions[idx].split('_')[-1]))
                                tension_original_changed_diffs.append(tension_diff_original_changed)

                            tension_diff_generated_changed = abs(int(tensions[mask_bar]) - int(changed_tensions[idx].split('_')[-1]))

                            # other_tension_diff_original = abs(
                            #     int(other_original_tensions[idx].split('_')[-1]) - int(tensions_other[mask_bar]))


                            tension_generated_changed_diffs.append(tension_diff_generated_changed)

                            # other_tension_original_diffs.append(other_tension_diff_original)

                            if tension_diff_generated_changed > 1:
                                new_mask_bars.append(mask_bar)
                            if generate_times == 0:
                                print(f'generated tension is {tensions[mask_bar]}, changed tension is {changed_tensions[idx]}, original tension is {original_tensions[idx]}')
                            else:
                                print(
                                    f'generated tension is {tensions[mask_bar]}, changed tension is {changed_tensions[idx]}')


                        original_changed_mean = np.mean(tension_original_changed_diffs)
                        generated_changed_mean = np.mean(tension_generated_changed_diffs)
                        better_ratio = np.sum(np.array(tension_generated_changed_diffs) <np.array(tension_original_changed_diffs)) / len(tension_generated_changed_diffs)
                        not_worse_ratio = np.sum(
                            np.array(tension_generated_changed_diffs) <= np.array(tension_original_changed_diffs)) / len(
                            tension_generated_changed_diffs)
                        # other_mean_diff = np.mean(other_tension_original_diffs)
                        if generate_times == 0:
                            tension_result_list = []
                            tension_result_list.append(original_changed_mean)
                            tension_result_list.append(generated_changed_mean)
                            tension_result_list.append(better_ratio)
                            tension_result_list.append(not_worse_ratio)
                        # else:
                        #     tension_result_list.append(generated_changed_mean)
                        # code for regenerate
                        # if len(new_mask_bars) > 0 and generate_times < 2:
                        #     mask_bars = new_mask_bars
                        #     regenerate = True
                        #     generate_times += 1
                        #     print(f'bar {mask_bars} to regenerate, total times {generate_times}')
                        #     batch = restored_with_generated_token
                        #
                        # else:
                        #     regenerate = False
                        #     if song_name in tension_dict:
                        #         tension_dict[song_name].append(tension_result_list)
                        #
                        #     else:
                        #         tension_dict[song_name] = [tension_result_list]
                        #
                        #
                        #     generate_times = 0
                        #     idx_number += 1
                        #     failed_times = 0
                        #
                        #     generated_pm.write(f'{output_folder}/change/generated_{song_name}_idx_{idx_number}.mid')
                        #     original_pm.write(f'{output_folder}/change/original_{song_name}_idx_{idx_number}.mid')

                        if song_name in tension_dict:
                            tension_dict[song_name].append(tension_result_list)

                        else:
                            tension_dict[song_name] = [tension_result_list]

                        generate_times = 0
                        idx_number += 1
                        failed_times = 0

                        generated_pm.write(f'{output_folder}/change/generated_{song_name}_idx_{idx_number}.mid')
                        original_pm.write(f'{output_folder}/change/original_{song_name}_idx_{idx_number}.mid')

                if control_number in [4, 5, 6]:
                    idx_number += 1
                    failed_times = 0
                    r = re.compile('i_\d')

                    generated_track_control = cal_track_control(restored_with_generated_token,
                                                                generated_pm)

                    original_track_control = cal_track_control(restored_with_target_token,
                                                               original_pm)

                    track_program = list(filter(r.match, batch))
                    track_nums = len(track_program)


                    for i in range(0, len(generated_track_control), track_nums):
                        for track_num in range(track_nums):
                            if track_num in mask_track_names:

                                if selected_control_name == 'polyphony':
                                    compare_name = 'y'
                                if selected_control_name == 'density':
                                    compare_name = 'd'
                                if selected_control_name == 'occupation':
                                    compare_name = 'o'

                                if original_track_control[i + track_num][0] == compare_name:
                                    print(
                                        f' target track changed control {track_num}: {(original_track_control[i + track_num], generated_track_control[i + track_num])}')

                                    if selected_control_name[0] == 'd':

                                        changed_track_original_control_diff_dict[mask_tracks[0]]['density'].append(original_control_diff)


                                        changed_track_control_generated_diff_dict[mask_tracks[0]]['density'].append(
                                            int(new_track_control[-1]) - int(generated_track_control[i + track_num][-1]))

                                        changed_generated_track_dict[mask_tracks[0]]['density'].append(int(
                                                generated_track_control[i + track_num][-1]))



                                    elif selected_control_name[0] == 'o':
                                        changed_track_original_control_diff_dict[mask_tracks[0]]['occupation'].append(
                                            original_control_diff)

                                        changed_track_control_generated_diff_dict[mask_tracks[0]]['occupation'].append(
                                            int(new_track_control[-1]) - int(
                                                generated_track_control[i + track_num][-1]))

                                        changed_generated_track_dict[mask_tracks[0]]['occupation'].append(int(
                                            generated_track_control[i + track_num][-1]))



                                    elif selected_control_name[0] == 'p':
                                        changed_track_original_control_diff_dict[mask_tracks[0]]['polyphony'].append(
                                            original_control_diff)

                                        changed_track_control_generated_diff_dict[mask_tracks[0]]['polyphony'].append(
                                            int(new_track_control[-1]) - int(
                                                generated_track_control[i + track_num][-1]))

                                        changed_generated_track_dict[mask_tracks[0]]['polyphony'].append(int(
                                            generated_track_control[i + track_num][-1]))

                                    else:
                                        pass

                                else:
                                    print(
                                        f' target track other control {track_num}: {(original_track_control[i + track_num], generated_track_control[i + track_num])}')

                                    if original_track_control[i + track_num][0] == 'd':

                                        changed_track_other_diff_dict[mask_tracks[0]]['density'].append(
                                            int(original_track_control[i + track_num][-1]) - int(
                                                generated_track_control[i + track_num][-1]))

                                    elif original_track_control[i + track_num][0] == 'o':
                                        changed_track_other_diff_dict[mask_tracks[0]]['occupation'].append(
                                            int(original_track_control[i + track_num][-1]) - int(
                                                generated_track_control[i + track_num][-1]))

                                    elif original_track_control[i + track_num][0] == 'y':
                                        changed_track_other_diff_dict[mask_tracks[0]]['polyphony'].append(
                                            int(original_track_control[i + track_num][-1]) - int(
                                                generated_track_control[i + track_num][-1]))

                                    else:

                                        pass

        if not bar_mode:

            total_generated_pm = remi_2midi(total_generated_tokens)
            total_original_pm = remi_2midi(total_original_tokens)


            if control_phase == 'original':
                total_original_pm.write(f'{output_folder}/original/original_{song_name}.mid')
            else:
                total_original_pm.write(f'{output_folder}/change/original_{song_name}.mid')



            if control_phase == 'original':
                total_generated_pm.write(f'{output_folder}/original/generated_{song_name}_track_{mask_track_names[0]}.mid')
            else:
                total_generated_pm.write(f'{output_folder}/change/generated_{song_name}_change_track_{mask_track_names[0]}.mid')


sys.exit()




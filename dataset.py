import random
import torch
import gc
from torch.utils.data import Dataset
from preprocessing import event_2midi,midi_2event
from preprocessing import remove_control_event
from preprocessing import remove_empty_track
from einops import rearrange
import re
import json
import os
from vocab import *
import sys
import tension_calculation
from joblib import Parallel, delayed


class ParallelLanguageDataset(Dataset):
    def __init__(self, event_folder,
                 tension_folder,
                 vocab, start_ratio,
                 end_ratio,
                 max_token_length,
                 window_size,
                 batches,batch_lengths,batch_size,
                 total_mask_ratio, structure_mask_ratio,
                 duration_mask_ratio, pitch_mask_ratio,
                 control_mask_ratio, header_mask_ratio,
                 ignore_ratio,
                 span_lengths,
                 span_ratio_jointly,
                 span_ratio_separately_each_epoch,
                 logger,
                 mask_bar_num_ratio,
                 mask_track_num_ratio,
                 mask_bar_ctrl_token=False,
                 pretraining=True,
                 fine_tune_prediction=False,
                 train_jointly=True,
                 verbose=False
                 ):
        """
        Initializes the dataset
                Parameters:
                        data_path_1 (str): Path to the English pickle file processed in process-data.py
                        data_path_2 (str): Path to the French pickle file processed in process-data.py
                        num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                        max_seq_length (int): Maximum number of tokens in each sentence pair
        """
        self.event_folder = event_folder
        self.tension_folder = tension_folder
        self.vocab = vocab
        self.batch_size = batch_size
        self.verbose = verbose
        self.logger = logger
        # print(f'current folder is {os.getcwd()}')
        # print(f'event folder {self.event_folder}')
        # self.files = self.walk(self.event_folder)

        # start_idx = int(len(self.files) * start_ratio)
        # end_idx = int(len(self.files) * end_ratio)
        # print(f'start idx is {start_idx}')
        # print(f'end idx is {end_idx}')
        # self.files = self.files[start_idx:end_idx]
        # self.keydata = json.load(open(tension_folder + '/files_result.json','r'))
        # self.batches, self.batch_lengths = self.gen_batches(self.files,self.keydata, max_token_length, window_size)
        # file_ratio = end_ratio - start_ratio
        # self.key_data = key_data,
        self.batches = batches,
        self.batches = self.batches[0]
        self.batch_lengths = batch_lengths,
        self.batch_lengths = self.batch_lengths[0]
        self.total_mask_ratio = total_mask_ratio
        self.structure_mask_ratio = structure_mask_ratio
        self.duration_mask_ratio = duration_mask_ratio
        self.pitch_mask_ratio = pitch_mask_ratio
        self.control_mask_ratio = control_mask_ratio
        self.header_mask_ratio = header_mask_ratio
        self.ignore_ratio = ignore_ratio
        self.span_lengths = span_lengths
        self.span_ratio_jointly = span_ratio_jointly
        self.span_ratio_separately_each_epoch = span_ratio_separately_each_epoch

        self.epoch = 0
        self.previous_index = 0
        self.mask_bar_num_ratio = mask_bar_num_ratio,
        self.mask_track_num_ratio = mask_track_num_ratio,
        self.mask_bar_ctrl_token = mask_bar_ctrl_token,
        self.train_jointly = train_jointly
        self.pretraining = pretraining
        self.fine_tuning_prediction = fine_tune_prediction

        print(f'pretraining is {self.pretraining}')
        if self.pretraining:
            print(f'control mask ratio is {self.control_mask_ratio}')
        else:
            if self.fine_tuning_prediction:
                print("mask track control tokens")
            else:
                print(f'mask_bar_num_ratio is {self.mask_bar_num_ratio}')
                print(f'mask_track_num_ratio is {self.mask_track_num_ratio}')
                print(f'mask_bar_ctrl_token is {self.mask_bar_ctrl_token}')





    def __getitem__(self, idx):

        if self.epoch > -1:
            if idx % self.batch_size == 0:
                this_idx = random.randint(0,len(self.batches)-1)
                if this_idx + self.batch_size - 1 > len(self.batches) - 1:
                    this_idx = this_idx - self.batch_size + 1
                self.previous_index = this_idx
            else:
                self.previous_index += 1
                this_idx = self.previous_index
        else:
            this_idx = idx

        if this_idx > len(self.batches) - 1:
            print(f'invalid this index {this_idx}')
            print(f'idx is {idx}')
            this_idx = len(self.batches) - 1
        #
        length = len(self.batches[this_idx])
        return_idx = random.choice(self.batch_lengths[length])

        event = self.batches[return_idx]
        # event = self.batches[this_idx]
        self.logger.debug((this_idx,return_idx))


        if self.pretraining:
            masked_input, decoder_in, decoder_target = self.random_word(event,
                                                                        self.total_mask_ratio,
                                                                        self.ignore_ratio,
                                                                        self.span_lengths,
                                                                        self.span_ratio_jointly,
                                                                        self.span_ratio_separately_each_epoch,
                                                                        self.epoch,
                                                                        self.train_jointly
                                                                        )

        # elif self.fine_tuning_prediction:
        #     masked_input, decoder_in, decoder_target = self.mask_category(event,control_tokens,3)
        #
        # else:
        #     masked_input, decoder_in, decoder_target = self.mask_bars(event,bar_ratio=self.mask_bar_num_ratio[0],
        #                                                             track_ratio=self.mask_track_num_ratio[0],
        #                                                             mask_bar_ctr=self.mask_bar_ctrl_token[0])
        #
        else:
            if random.random() > .8:
                result = self.mask_category(event,all_meta_tokens)
                if result is None:
                    return None
                else:
                    masked_input, decoder_in, decoder_target = result

            else:
                masked_input, decoder_in, decoder_target = self.mask_bars(event,bar_ratio=self.mask_bar_num_ratio[0],
                                                                        track_ratio=self.mask_track_num_ratio[0],
                                                                        mask_bar_ctr=self.mask_bar_ctrl_token[0])



        # masked_input, decoder_in, decoder_target = self.mask_category(event,
        #                                                             'pitch',20)
        if idx == len(self.batches) - 1:
            self.epoch += 1
            print(f'epoch is {self.epoch}')

        return masked_input, decoder_in, decoder_target

    def __len__(self):
        return len(self.batches)

    def gen_batches(self, files, key_data, max_token_length=3000, batch_window_size=8):

        batches = []

        for i in range(len(files)):
            file_events = np.array(pickle.load(open(files[i], 'rb')))
            num_of_tracks = len(file_events[3:np.where('track_0' == file_events)[0][0]])
            if num_of_tracks < 2:
                print(f'omit file {files[i]} with only one track')
                continue


            file_name_in_folder = files[i].split('lmd_separate_event')[1:][0][:-6]
            tensile_file = self.tension_folder + file_name_in_folder + '.tensile'
            diameter_file = self.tension_folder + file_name_in_folder + '.diameter'

            tensiles = np.array(pickle.load(open(tensile_file, 'rb')))
            diameters = np.array(pickle.load(open(diameter_file, 'rb')))
            if self.tension_folder + file_name_in_folder + '.mid' in key_data:
                keys = key_data[self.tension_folder + file_name_in_folder + '.mid']
            else:
                print(f'omit file {files[i]} with no key')
                continue
            # if keys[2] != -1:
            #     print(f'file name is {files[i]}')
            bar_pos = np.where(file_events == 'bar')[0]

            total_bars = min(len(tensiles), len(diameters), len(bar_pos))
            if total_bars < len(bar_pos):
                bar_pos = bar_pos[:total_bars+1]
                file_events = file_events[:bar_pos[-1]]
                bar_pos = bar_pos[:-1]

            result = add_control_event(file_events, bar_pos, tensiles, diameters,keys)
            if result:
                events_with_control, keys = result
                bar_pos = np.where(events_with_control == 'bar')[0]
                # total_bars = min(len(tensiles), len(diameters), len(bar_pos))
                # bar_pos = bar_pos[:total_bars]

                bar_beginning_pos = bar_pos[::batch_window_size]

                meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0]-2]

                for pos in range(len(bar_beginning_pos) - 1):

                    # print(bar_beginning_pos[pos])
                    if keys[2] != -1 and pos*8+1 >= keys[2]:
                        meta_events[2] = key_to_token[keys[3]]
                    if pos == len(bar_beginning_pos) - 2 and pos != 0:
                        # skip the last one
                        # continue
                        # return_events = file_events[bar_beginning_pos[pos]:]
                        return_events = np.insert(events_with_control[bar_beginning_pos[pos]:], 1, meta_events)
                    elif pos > 0:
                        return_events = np.insert(events_with_control[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]], 1,
                                                  meta_events)
                    else:
                        return_events = events_with_control[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]]
                    batches.append(return_events.tolist())
        batches.sort(key=len)
        i = 0
        while i < len(batches) - 1:
            if batches[i] == batches[i + 1]:
                del batches[i + 1]
            else:
                i += 1

        batches_new = []
        this_batch_total_length = 0

        while len(batches) > 0:
            if this_batch_total_length + len(batches[0]) < max_token_length:
                if len(batches_new) > 0:
                    batches_new[-1].append(batches[0])
                else:
                    batches_new.append([batches[0]])
                this_batch_total_length += len(batches[0])
            else:
                if len(batches[0]) > max_token_length:
                    print(
                        f'the event size {len(batches[0])} is greater than {max_token_length}, skip this file, or increase the max token length')
                    this_batch_total_length = 0
                else:
                    batches_new.append([batches[0]])
                    this_batch_total_length = len(batches[0])
            del batches[0]
        del batches
        gc.collect()
        batch_lengths = {}
        for index, item in enumerate(batches_new):
            if len(item) not in batch_lengths:
                batch_lengths[len(item)] = [index]
            else:
                batch_lengths[len(item)].append(index)
        return batches_new, batch_lengths

    def random_word(self,
                    events,
                    total_ratio,
                    ignore_ratio,
                    span_lengths,
                    span_ratio_jointly,
                    span_ratio_separately_each_epoch,
                    epoch,
                    train_jointly=True):

        # ratios = [structure_ratio,duration_ratio, pitch_ratio,
        #           control_ratio, self.header_mask_ratio]
        # token_types = [structure_token, durations, pitches, control_tokens, header_token]

        total_tokens = []
        total_decoder_in = []
        total_decoder_target = []
        if train_jointly:
            # span_lengths = np.arange(1,span_lengths)
            # span_ratio_jointly = np.repeat(1/span_lengths,span_lengths)

            if span_lengths == 1:
                span_lengths = [1,2,3]
            elif span_lengths == 2:
                span_lengths = [2,1,3]
            else:
                span_lengths = [3,1,2]
            # elif span_lengths == 5:
            #     span_lengths = [5,4,3,2,1]
            # else:
            #     # == 8
            #     span_lengths = [8, 5, 6, 2, 1]
            #
            if span_ratio_jointly == 1:
                span_ratio_jointly = [1, 0, 0]
            else:
                span_ratio_jointly = [.5, .25, .25]
            # print(f'product is {np.dot(span_ratio_jointly,span_lengths)}')
            # print(f'product shape is {np.dot(span_ratio_jointly,span_lengths).shape}')
            random_threshold = total_ratio / (np.dot(span_ratio_jointly, span_lengths))
            span_ratio = span_ratio_jointly
        else:
            if span_lengths == 1:
                span_lengths = [1,2,3]
            elif span_lengths == 2:
                span_lengths = [2,1,3]
            else:
                span_lengths = [3,1,2]

            random_threshold = total_ratio / (np.dot(span_ratio_separately_each_epoch, np.array(span_lengths)))
            # print(f'random thresh is {random_threshold}')
            # print(f'random thresh shape is {random_threshold.shape}')
            random_threshold = random_threshold[epoch]
            span_ratio = span_ratio_separately_each_epoch[epoch]
        random.shuffle(events)
        for event in events:
            tokens = []
            decoder_in = []
            decoder_target = []
            start_pos = 0
            total_masked_ratio = 0
            total_control_tokens = 0

            control_masked_ratio = {}
            # for name in self.vocab.control_names:
            #     control_masked_ratio[name] = 0

            control_masked_ratio['total'] = 0

            masked_num = 0
            # bar_pos = np.where(np.array(event) == 'bar')[0]


            # print(len(event))
            # if self.epoch < -1:
            #     while total_masked_ratio < total_ratio and start_pos < len(event):
            #         masked_token = []
            #         prob = random.random()
            #
            #
            #         if prob < random_threshold:
            #             prob /= random_threshold
            #
            #             if prob < span_ratio[0]:
            #                 if start_pos + span_lengths[0] <= len(event):
            #                     masked_token = event[start_pos:start_pos + span_lengths[0]]
            #                     tokens.append(self.vocab.mask_indices[masked_num])
            #                     total_masked_ratio += span_lengths[0] / len(event)
            #                     start_pos += span_lengths[0]
            #             elif span_ratio[0] < prob < span_ratio[1] + span_ratio[0]:
            #                 if start_pos + span_lengths[1] <= len(event):
            #                     masked_token = event[start_pos:start_pos + span_lengths[1]]
            #                     tokens.append(self.vocab.mask_indices[masked_num])
            #                     total_masked_ratio += span_lengths[1] / len(event)
            #                     start_pos += span_lengths[1]
            #             else:
            #                 if start_pos + span_lengths[2] <= len(event):
            #                     masked_token = event[start_pos:start_pos + span_lengths[2]]
            #                     tokens.append(self.vocab.mask_indices[masked_num])
            #                     total_masked_ratio += span_lengths[2] / len(event)
            #                     start_pos += span_lengths[2]
            #
            #             if len(masked_token) > 0:
            #                 if not isinstance(masked_token, list):
            #                     masked_token = [masked_token]
            #                 decoder_in.append(self.vocab.mask_indices[masked_num])
            #                 for token in masked_token:
            #                     decoder_in.append(self.vocab.char2index(token))
            #                     decoder_target.append(self.vocab.char2index(token))
            #                 else:
            #                     decoder_target.append(self.vocab.eos_index)
            #
            #         else:
            #             tokens.append(self.vocab.char2index(event[start_pos]))
            #             start_pos += 1
            # else:
            # for token in event:
            #     if self.vocab.char2index(token) in self.vocab.control_indices:
            #         total_control_tokens += 1

            while total_masked_ratio < total_ratio and start_pos < len(event):
                masked_token = []
                prob = random.random()
                # have_control_token = False
                control_token_length = 0
                if prob < span_ratio[0]:
                    if start_pos + span_lengths[0] <= len(event):
                        # for event_token in event[start_pos:start_pos + 1]:
                        #     if self.vocab.char2index(event_token) in self.vocab.control_indices:
                        #         have_control_token = True
                        #         control_token_length += 1
                        # # all the control token mask span length = 1
                        # if have_control_token:
                        #     prob = random.random()
                        #     if prob < self.control_mask_ratio:
                        #         masked_token = event[start_pos:start_pos + 1]
                        #         tokens.append(self.vocab.mask_indices[masked_num])
                        #         total_masked_ratio += 1 / len(event)
                        #         control_masked_ratio['total'] += control_token_length / total_control_tokens
                        #         start_pos += 1
                        # else:
                        prob = random.random()
                        if prob < random_threshold * 1.5:
                            masked_token = event[start_pos:start_pos + span_lengths[0]]
                            tokens.append(self.vocab.mask_indices[masked_num])
                            total_masked_ratio += span_lengths[0] / len(event)
                            start_pos += span_lengths[0]

                elif span_ratio[0] < prob < span_ratio[1] + span_ratio[0]:
                    if start_pos + span_lengths[1] <= len(event):
                        # for event_token in event[start_pos:start_pos + 1]:
                        #     if self.vocab.char2index(event_token) in self.vocab.control_indices:
                        #         have_control_token = True
                        #         control_token_length += 1
                        #
                        # if have_control_token:
                        #     prob = random.random()
                        #     if prob < self.control_mask_ratio:
                        #         masked_token = event[start_pos:start_pos + 1]
                        #         tokens.append(self.vocab.mask_indices[masked_num])
                        #         control_masked_ratio['total'] += control_token_length / total_control_tokens
                        #         total_masked_ratio += 1 / len(event)
                        #         start_pos += 1
                        # else:
                        prob = random.random()
                        if prob < random_threshold * 1.5:
                            masked_token = event[start_pos:start_pos + span_lengths[1]]
                            tokens.append(self.vocab.mask_indices[masked_num])
                            total_masked_ratio += span_lengths[1] / len(event)
                            start_pos += span_lengths[1]
                else:
                    if start_pos + span_lengths[2] <= len(event):
                        # for event_token in event[start_pos:start_pos + 1]:
                        #     if self.vocab.char2index(event_token) in self.vocab.control_indices:
                        #         have_control_token = True
                        #         control_token_length += 1
                        #
                        # if have_control_token:
                        #     prob = random.random()
                        #     if prob < self.control_mask_ratio:
                        #         masked_token = event[start_pos:start_pos + 1]
                        #         tokens.append(self.vocab.mask_indices[masked_num])
                        #         control_masked_ratio['total'] += control_token_length / total_control_tokens
                        #         total_masked_ratio += 1 / len(event)
                        #         start_pos += 1
                        # else:
                        prob = random.random()
                        if prob < random_threshold * 1.5:
                            masked_token = event[start_pos:start_pos + span_lengths[2]]
                            tokens.append(self.vocab.mask_indices[masked_num])
                            total_masked_ratio += span_lengths[2] / len(event)
                            start_pos += span_lengths[2]

                if len(masked_token) > 0:
                    if not isinstance(masked_token, list):
                        masked_token = [masked_token]
                    decoder_in.append(self.vocab.mask_indices[masked_num])
                    for token in masked_token:
                        decoder_in.append(self.vocab.char2index(token))
                        decoder_target.append(self.vocab.char2index(token))
                    else:
                        decoder_target.append(self.vocab.eos_index)

                else:
                    tokens.append(self.vocab.char2index(event[start_pos]))
                    start_pos += 1


            while start_pos < len(event):
                tokens.append(self.vocab.char2index(event[start_pos]))
                start_pos += 1

            # add ignore token randomly
            for i,token in enumerate(tokens):
                if token != self.vocab.mask_indices[0]:
                    if random.random() < ignore_ratio:
                        tokens[i] = self.vocab.ignore_indices[0]

            tokens = np.array(tokens)
            if len(decoder_in) > 0:
                decoder_in = np.array(decoder_in)
                decoder_target = np.array(decoder_target)
                total_tokens.append(tokens)
                total_decoder_in.append(decoder_in)
                total_decoder_target.append(decoder_target)

                # debug purpose
                # print('\n')
                # print(f'event length is {len(event)}')
                # print(f'tokens length is {len(tokens)}')
                # print(f'control tokens length is {total_control_tokens}')
                # print(f'masked ratio is {total_masked_ratio}')
                # print(f'control masked ratio is {control_masked_ratio["total"]}')
                # print(f'decoder_in length is {len(decoder_in)}')
                # print(f'decoder_out length is {len(decoder_target)}')
                # print(f'ratio is {(len(tokens) + len(decoder_in)) / len(event)}')

        # print(len(tokens) - len(np.where(output_label==2)[0]))
        # print(len(output_label) - len(np.where(output_label==2)[0])*2)
        return total_tokens, total_decoder_in, total_decoder_target

    def mask_category(self, events, token_type):

        total_tokens = []
        total_decoder_in = []
        total_decoder_target = []

        token_lengths = []
        event_lengths = []
        decoder_in_lengths = []
        decoder_out_lengths = []
        total_lengths_list = []


        random.shuffle(events)
        total_lengths = 0
        for event in events:
            tokens = []
            decoder_in = []
            decoder_target = []
            start_pos = 0
            total_masked_ratio = 0
            masked_num = 0
            in_event = False
            token_type_pos = []
            track_control_pos = []
            bar_control_pos = []
            song_control_pos = []
            song_without_key_pos = []

            for pos,token in enumerate(event):
                if token in token_type:
                    token_type_pos.append(pos)
                if token in track_control_tokens:
                    track_control_pos.append(pos)
                if token in bar_control_tokens:
                    bar_control_pos.append(pos)
                if token in header_token:
                    song_control_pos.append(pos)
                if token in header_without_key_token:
                    song_without_key_pos.append(pos)



            # mask_number = np.random.randint(int(len(token_type_pos)/2), len(token_type_pos))
            #
            # # print(f'mask_number is {mask_number}')
            # # print(f'token_type_pos is {token_type_pos}')
            # mask_poses = np.sort(np.random.choice(token_type_pos, size=mask_number+1, replace=False))
            #
            #
            # while start_pos < len(event):
            #
            #
            #     if start_pos in mask_poses:
            #
            #         masked_token = event[start_pos]
            #         tokens.append(self.vocab.mask_indices[0])
            #
            #         decoder_target.append(self.vocab.char2index(masked_token))
            #         decoder_target.append(self.vocab.eos_index)
            #
            #         decoder_in.append(self.vocab.mask_indices[0])
            #
            #         decoder_in.append(self.vocab.char2index(masked_token))
            #
            #         masked_num += 1
            #
            #     else:
            #         tokens.append(self.vocab.char2index(event[start_pos]))
            #     start_pos += 1
            #
            # while start_pos < len(event):
            #     tokens.append(self.vocab.char2index(event[start_pos]))
            #     start_pos += 1
            #
            # # only choose one control to mask
            # pos_chosen = np.random.choice(token_type_pos,1)[0]
            #
            # masked_token = event[pos_chosen]
            # tokens[pos_chosen] = self.vocab.mask_indices[0]
            # decoder_target.append(self.vocab.char2index(masked_token))
            # decoder_target.append(self.vocab.eos_index)
            #
            # decoder_in.append(self.vocab.mask_indices[0])
            # decoder_in.append(self.vocab.char2index(masked_token))
            #
            #
            #
            # # random ignore 50% of the track token in a different bar/track
            # bar_poses = np.where(np.array(event) == 'bar')[0]
            #
            # r = re.compile('i_\d')
            #
            # track_program = list(filter(r.match, event))
            # track_nums = len(track_program)
            #
            #
            # track_end_poses = []
            # if track_nums == 3:
            #     #         ratios = track_ratio[0]
            #
            #     track_0_pos = np.where('track_0' == np.array(event))[0]
            #     track_1_pos = np.where('track_1' == np.array(event))[0]
            #     track_2_pos = np.where('track_2' == np.array(event))[0]
            #     for pos in track_2_pos[:-1]:
            #         track_end_poses.append(bar_poses[np.where(pos < bar_poses)[0][0]])
            #     else:
            #         track_end_poses.append(len(event))
            #     all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos, track_2_pos, track_end_poses]))
            #
            # elif track_nums == 2:
            #     #         ratios = track_ratio[1]
            #     track_0_pos = np.where('track_0' == np.array(event))[0]
            #     track_1_pos = np.where('track_1' == np.array(event))[0]
            #     for pos in track_1_pos[:-1]:
            #         track_end_poses.append(bar_poses[np.where(pos < bar_poses)[0][0]])
            #     else:
            #         track_end_poses.append(len(event))
            #     all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos, track_end_poses]))
            #
            # else:
            #     track_0_pos = np.where('track_0' == np.array(event))[0]
            #     for pos in track_0_pos[:-1]:
            #         track_end_poses.append(bar_poses[np.where(pos < bar_poses)[0][0]])
            #     else:
            #         track_end_poses.append(len(event))
            #     all_track_pos = np.sort(np.concatenate([track_0_pos, track_end_poses]))
            #
            # bar_with_track_poses = []
            #
            #
            # for i, pos in enumerate(all_track_pos):
            #     if i % (track_nums + 1) == 0:
            #         this_bar_poses = []
            #         this_bar_pairs = []
            #         this_bar_poses.append(pos)
            #
            #     else:
            #         this_bar_poses.append(pos)
            #         if i % (track_nums + 1) == track_nums:
            #             for j in range(len(this_bar_poses) - 1):
            #                 this_bar_pairs.append((this_bar_poses[j], this_bar_poses[j + 1]))
            #
            #             bar_with_track_poses.append(this_bar_pairs)
            #
            #
            # if pos_chosen in track_control_pos:
            #     # random ignore 50% other track controls and bar controls
            #     for i,token_idx in enumerate(tokens):
            #         if i in all_meta_tokens:
            #             if random.random() > .5 and i != pos_chosen:
            #                 tokens[i] = self.vocab.ignore_indices[0]
            #
            #     if track_nums > 1:
            #         # mask 50% irrelevant tracks
            #         selected_track = np.where(pos_chosen == track_control_pos)[0][0] % track_nums
            #
            #         # random ignore 50% of the bar track tokens
            #         selected_ignore_poses = []
            #         for bar_num, track_poses in enumerate(bar_with_track_poses):
            #             for track_num,pos_pair in enumerate(track_poses):
            #                 if track_num != selected_track:
            #                     if random.random() > .5:
            #                         selected_ignore_poses.append(pos_pair)
            #
            #         for pair in selected_ignore_poses[::-1]:
            #             # logger.info(masked_pairs)
            #
            #                 # logger.info(token_events[masked_pairs[0]:masked_pairs[1]])
            #             for pop_time in range(pair[1] - pair[0]):
            #                 tokens.pop(pair[0])
            #             tokens.insert(pair[0], self.vocab.ignore_indices[0])
            #
            # if pos_chosen in bar_control_pos:
            #
            #     # random ignore 50% other controls,other bars but keep key control
            #     for i, token_idx in enumerate(tokens):
            #         if i in track_control_pos or i in bar_control_pos or i in song_without_key_pos:
            #             if random.random() > .5 and i != pos_chosen:
            #                 tokens[i] = self.vocab.ignore_indices[0]
            #
            #     selected_bar = int(np.where(pos_chosen == bar_control_pos)[0][0] / 2)
            #
            #     #random ignore 50% of the bar track tokens
            #     selected_ignore_poses = []
            #     for bar_num, track_poses in enumerate(bar_with_track_poses):
            #         if bar_num != selected_bar:
            #             if random.random() > .5:
            #                 selected_ignore_poses.append(bar_with_track_poses[bar_num])
            #
            #     for ignore_poses in selected_ignore_poses[::-1]:
            #         # logger.info(masked_pairs)
            #         for pair in ignore_poses[::-1]:
            #         # logger.info(token_events[masked_pairs[0]:masked_pairs[1]])
            #             for pop_time in range(pair[1] - pair[0]):
            #                 tokens.pop(pair[0])
            #             tokens.insert(pair[0], self.vocab.ignore_indices[0])
            #     print(tokens)
            #
            #
            #     token_words = []
            #     for token in tokens:
            #         token_words.append(self.vocab._idx2char[token])
            #


            # mask_number = np.random.randint(1,number+1)

            # mask_pos = np.random.choice(token_type_pos,mask_number)
            # mask_pos = np.random.choice(token_type_pos, len(token_type_pos))

            mask_number = np.random.randint(0, len(token_type_pos))
            mask_poses = np.sort(np.random.choice(token_type_pos, size=mask_number, replace=False))

            left_poses = np.setdiff1d(token_type_pos, mask_poses)
            ignore_number = np.random.randint(0, len(left_poses))
            ignore_poses = np.sort(np.random.choice(left_poses, size=ignore_number, replace=False))

            while start_pos < len(event):

                #random select ratio of mask and ignore tokens

                # original_poses = np.setdiff1d(left_poses,ignore_poses)

                if start_pos in mask_poses:

                    masked_token = event[start_pos]
                    tokens.append(self.vocab.mask_indices[0])

                    decoder_target.append(self.vocab.char2index(masked_token))
                    decoder_target.append(self.vocab.eos_index)

                    decoder_in.append(self.vocab.mask_indices[0])
                    # use ignore token here
                    if masked_token in key_token:
                        decoder_in.append(self.vocab.char2index(masked_token))
                    else:
                        decoder_in.append(self.vocab.ignore_indices[0])

                    masked_num += 1
                elif start_pos in ignore_poses:
                    tokens.append(self.vocab.ignore_indices[0])
                # elif start_pos in original_poses:
                #     tokens.append(self.vocab.char2index(event[start_pos]))
                else:
                    tokens.append(self.vocab.char2index(event[start_pos]))
                start_pos += 1


            tokens = np.array(tokens)
            if len(decoder_in) > 0:
                decoder_in = np.array(decoder_in)
                decoder_target = np.array(decoder_target)

                total_tokens.append(tokens)
                total_decoder_in.append(decoder_in)
                total_decoder_target.append(decoder_target)

                this_total_length = len(tokens) + len(decoder_in) + len(decoder_target)
                # self.logger.info(f'this total lengths is {this_total_length}')
                total_lengths += this_total_length

                token_lengths.append(len(tokens))
                event_lengths.append(len(event))
                decoder_in_lengths.append(len(decoder_in))
                decoder_out_lengths.append(len(decoder_target))
                total_lengths_list.append(this_total_length)

                # self.logger.info(f'event lengths is {np.sum(np.array(event_lengths))}')

        # if total_lengths > 4300:
        #     self.logger.info(f'one batch total length is {total_lengths}')
        #     self.logger.info(f'event lengths is {np.sum(np.array(event_lengths))}')
        #     self.logger.info(f'token lengths is {token_lengths}')
        #     # self.logger.info(f'decoder in lengths is {decoder_in_lengths}')
        #     self.logger.info(f'decoder out lengths is {decoder_out_lengths}')
        #     self.logger.info(f'ratio is {total_lengths/np.sum(np.array(event_lengths))}')

        # print(len(tokens) - len(np.where(output_label==2)[0]))
        # print(len(output_label) - len(np.where(output_label==2)[0])*2)
        if len(total_tokens) == 0:
            return None

        return total_tokens, total_decoder_in, total_decoder_target


    def mask_bars(self,events,bar_ratio,track_ratio,mask_bar_ctr=False):

        # mask bar token (w/wo bar control token) and try to generate bar token

        total_tokens = []
        total_decoder_in = []
        total_decoder_target = []

        token_lengths = []
        event_lengths = []
        decoder_in_lengths = []
        decoder_out_lengths = []
        total_lengths_list = []

        random.shuffle(events)


        if random.random() > 0.5:
            mask_mode = 0
        elif random.random() > 0.25:
            mask_mode = 1
        else:
            mask_mode = 2

        # self.logger.info(f'mask mode is {mask_mode}')
        total_lengths = 0
        for event in events:
            tokens = []
            decoder_in = []
            decoder_target = []
            masked_indices_pairs = []

            # while start_pos < len(event):
            #     tokens.append(self.vocab.char2index(event[start_pos]))
            #     start_pos += 1



            bar_poses = np.where(np.array(event) == 'bar')[0]

            r = re.compile('i_\d')

            track_program = list(filter(r.match, event))
            track_nums = len(track_program)
            track_end_poses = []
            if track_nums == 3:
                #         ratios = track_ratio[0]

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
                        bar_with_track_poses[bar_mask_pos][track_mask_pos]
                        masked_indices_pairs.append(bar_with_track_poses[bar_mask_pos][track_mask_pos])
            elif mask_mode == 1:
                # mask whole tracks
                track_mask_number = np.random.randint(0, track_nums)
                track_mask_poses = np.sort(np.random.choice(track_nums, size=track_mask_number + 1, replace=False))
                for bar_num, tracks_in_a_bar in enumerate(bar_with_track_poses):
                    for track_pos, track_star_end_poses in enumerate(tracks_in_a_bar):
                        if track_pos in track_mask_poses:
                            masked_indices_pairs.append(track_star_end_poses)

            else:
                # mask whole bars
                bar_mask_number = np.random.randint(0, len(bar_poses))
                bar_mask_poses = np.sort(np.random.choice(len(bar_poses),size=bar_mask_number+1,replace=False))

                for bar_mask_pos in bar_mask_poses:
                    for tracks_in_a_bar in bar_with_track_poses[bar_mask_pos]:
                        masked_indices_pairs.append(tracks_in_a_bar)


            token_events = event.copy()

            for masked_pairs in masked_indices_pairs:
                masked_token = event[masked_pairs[0]:masked_pairs[1]]
                # print(masked_token)
                decoder_in.append(self.vocab.mask_indices[0])
                for token in masked_token:
                    decoder_in.append(self.vocab.char2index(token))
                    decoder_target.append(self.vocab.char2index(token))
                else:
                    decoder_target.append(self.vocab.eos_index)

            for masked_pairs in masked_indices_pairs[::-1]:
                # print(masked_pairs)
                # print(token_events[masked_pairs[0]:masked_pairs[1]])
                for pop_time in range(masked_pairs[1] - masked_pairs[0]):
                    token_events.pop(masked_pairs[0])
                token_events.insert(masked_pairs[0], mask[0])

            for token in token_events:
                tokens.append(self.vocab.char2index(token))

            tokens = np.array(tokens)
            if len(decoder_in) > 0:
                decoder_in = np.array(decoder_in)
                decoder_target = np.array(decoder_target)
                # print('\n')


                # self.logger.info(f'event length is {len(event)}')
                # self.logger.info(f'tokens length is {len(tokens)}')
                # print(f'masked num is {masked_num}')
                # self.logger.info(f'decoder_in length is {len(decoder_in)}')
                # self.logger.info(f'decoder_out length is {len(decoder_target)}')
                this_total_length = len(tokens) + len(decoder_in) + len(decoder_target)
                # self.logger.info(f'this total lengths is {this_total_length}')
                total_lengths += this_total_length

                # print(f'ratio is {(len(tokens) + len(decoder_in)) / len(event)}')
                total_tokens.append(tokens)
                total_decoder_in.append(decoder_in)
                total_decoder_target.append(decoder_target)

                token_lengths.append(len(tokens))
                event_lengths.append(len(event))
                decoder_in_lengths.append(len(decoder_in))
                decoder_out_lengths.append(len(decoder_target))
                total_lengths_list.append(this_total_length)

        if len(total_tokens) == 0:
            print('why')
        # if total_lengths > 4000:
        #     # self.logger.info(f'one batch total length is {total_lengths}')
        #     self.logger.info(f'event lengths is {event_lengths}')
        #     # self.logger.info(f'token lengths is {token_lengths}')
        #     # self.logger.info(f'decoder in lengths is {decoder_in_lengths}')
        #     # self.logger.info(f'decoder out lengths is {decoder_out_lengths}')
        #     self.logger.info(f'total lengths is {total_lengths}')
        #
        #     self.logger.info(f'mask mode is {mask_mode}')
            # total_tokens.pop()
            # total_decoder_in.pop()
            # total_decoder_target.pop()

        # print(len(tokens) - len(np.where(output_label==2)[0]))
        # print(len(output_label) - len(np.where(output_label==2)[0])*2)
        return total_tokens, total_decoder_in, total_decoder_target

    def shuffle_batches(self):
        self.batches = self.gen_batches(self.num_tokens, self.data_lengths)

    def walk(self, folder_name):
        files = []
        for p, d, f in os.walk(folder_name):
            for file_name in f:

                if file_name[-5:] == 'event':
                    files.append(os.path.join(p, file_name))
        return files


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')

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

def collate_mlm(batch):
    batch = list(filter(None, batch))
    if len(batch) == 0:
        return None
    max_input_len = max_target_len = 0
    for batch_dim in range(len(batch)):
        input_lens = [x.shape[0] for x in batch[batch_dim][0]]
        max_input_len = max(max_input_len,max(input_lens))

        target_lens = [x.shape[0] for x in batch[batch_dim][1]]
        max_target_len = max(max_target_len,max(target_lens))

    input_pad_list = []
    input_pad_masks_list = []
    target_in_pad_list = []
    target_in_pad_masks_list = []
    target_out_pad_list = []

    for batch_dim in range(len(batch)):

        # input
        input_padded = [pad1d(x, max_input_len) for x in batch[batch_dim][0]]
        input_padded = np.stack(input_padded)

        input_pad_masks = input_padded == 0

        # target
        target_in_padded = [pad1d(x, max_target_len) for x in batch[batch_dim][1]]
        target_in_padded = np.stack(target_in_padded)

        target_in_pad_masks = target_in_padded == 0

        target_out_padded = [pad1d(x, max_target_len) for x in batch[batch_dim][2]]
        target_out_padded = np.stack(target_out_padded)

        input_pad_list.append(input_padded)
        input_pad_masks_list.append(input_pad_masks)
        target_in_pad_list.append(target_in_padded)
        target_in_pad_masks_list.append(target_in_pad_masks)
        target_out_pad_list.append(target_out_padded)

    input_pad = torch.tensor(np.concatenate(input_pad_list)).long()
    target_in_pad = torch.tensor(np.concatenate(target_in_pad_list)).long()
    target_out_pad = torch.tensor(np.concatenate(target_out_pad_list)).long()
    input_pad_masks = torch.tensor(np.concatenate(input_pad_masks_list)).bool()
    target_in_pad_masks = torch.tensor(np.concatenate(target_in_pad_masks_list)).bool()


    output = {"input": input_pad,
              "target_in": target_in_pad,
              "target_out": target_out_pad,
              "input_pad_mask": input_pad_masks,
              "target_pad_mask": target_in_pad_masks
              }

    return output

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


#
def cal_tension(pm):


    result = tension_calculation.extract_notes(pm, 3)


    pm, piano_roll, sixteenth_time, beat_time, down_beat_time, beat_indices, down_beat_indices = result

    key_name = tension_calculation.all_key_names

    result = tension_calculation.cal_tension(
        piano_roll, beat_time, beat_indices, down_beat_time,
        down_beat_indices, -1, key_name)

    tensiles, diameters, key_name = result

    tensile_category = to_category(tensiles,tensile_bins)
    diameter_category = to_category(diameters, diameter_bins)

    # print(f'key is {key_name}')

    return tensile_category, diameter_category, key_name

def add_control_event(file_events,header_events):
    file_events = np.copy(file_events)
    num_of_tracks = len(header_events[2:])

    # if file_events[1] not in time_signature_token:
    #     file_events = np.insert(file_events,1,time_signature)
    #     file_events = np.insert(file_events, 2, tempo)
    #     for i, program in enumerate(header_events[2:]):
    #         file_events = np.insert(file_events, 3+i, program)


    for event in header_events[::-1]:
        file_events = np.insert(file_events, 0, event)

    bar_pos = np.where(file_events == 'bar')[0]
    pm = event_2midi(file_events.tolist())[0]
    pm = remove_empty_track(pm)
    if len(pm.instruments) < 1:
        return None

    tensiles,diameters,key = cal_tension(pm)

    if tensiles is not None:
        total_bars = min(len(tensiles), len(diameters), len(bar_pos))
        if total_bars < len(bar_pos):
            print(f'total bars is {total_bars}. less than original {len(bar_pos)}')
            bar_pos = bar_pos[:total_bars + 1]
            file_events = file_events[:bar_pos[-1]]
            bar_pos = bar_pos[:-1]

        if total_bars < len(tensiles):
            print(f'total bars is {total_bars}. less than tensile {len(tensiles)}')
            tensiles = tensiles[:total_bars]
            diameters = diameters[:total_bars]



    #     print(f'number of bars is {len(bar_pos)}')
    #     print(f'time signature is {file_event[1]}')
    bar_length = int(file_events[0][0])

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
            track_pos.append(np.where(track_name == bar_events)[0][0])
        #         print(track_pos)
        track_index = 0
        for track_index in range(len(track_names) - 1):
            track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
            track_events[track_names[track_index]].append(track_event)
        #             print(track_event)
        else:
            if track_index == 0:
                track_event = bar_events[track_pos[track_index]:]
                #             print(track_event)
                track_events[track_names[track_index]].append(track_event)
            else:
                track_index += 1
                track_event = bar_events[track_pos[track_index]:]
                #             print(track_event)
                track_events[track_names[track_index]].append(track_event)

    densities = note_density(track_events, bar_length)
    density_category = to_category(densities, control_bins)
    pm, _ = event_2midi(file_events.tolist())
    occupation_rate, polyphony_rate = occupation_polyphony_rate(pm)
    occupation_category = to_category(occupation_rate, control_bins)
    polyphony_category = to_category(polyphony_rate, control_bins)
    pitch_register_category = pitch_register(track_events)
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

    # print(track_control_tokens)

    file_events = file_events.tolist()



    key = key_to_token[key]
    file_events.insert(2, key)


    for token in track_control_tokens[::-1]:
        file_events.insert(3, token)

    if '_' not in file_events[1]:
        tempo = float(file_events[1])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        file_events[1] = f't_{tempo_category}'

    if tensiles is not None:

        tension_positions = np.where(np.array(file_events) == 'track_0')[0]

        total_insert = 0

        for i, pos in enumerate(tension_positions):
            file_events.insert(pos + total_insert, f's_{tensiles[i]}')
            total_insert += 1
            file_events.insert(pos + total_insert, f'a_{diameters[i]}')
            total_insert += 1

    return np.array(file_events)


# def gen_batches(num_tokens, data_lengths):
#     """
#      Returns the batched data
#              Parameters:
#                      num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
#                      data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
#                                          and values of the indices that correspond to these parallel sentences
#              Returns:
#                      batches (arr): List of each batch (which consists of an array of indices)
#      """
#
#     # Shuffle all the indices
#     for k, v in data_lengths.items():
#         random.shuffle(v)
#
#     batches = []
#     prev_tokens_in_batch = 1e10
#     for k in sorted(data_lengths):
#         # v contains indices of the sentences
#         v = data_lengths[k]
#         total_tokens = (k[0] + k[1]) * len(v)
#
#         # Repeat until all the sentences in this key-value pair are in a batch
#         while total_tokens > 0:
#             tokens_in_batch = min(total_tokens, num_tokens) - min(total_tokens, num_tokens) % (k[0] + k[1])
#             sentences_in_batch = tokens_in_batch // (k[0] + k[1])
#
#             # Combine with previous batch if it can fit
#             if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
#                 batches[-1].extend(v[:sentences_in_batch])
#                 prev_tokens_in_batch += tokens_in_batch
#             else:
#                 batches.append(v[:sentences_in_batch])
#                 prev_tokens_in_batch = tokens_in_batch
#             # Remove indices from v that have been added in a batch
#             v = v[sentences_in_batch:]
#
#             total_tokens = (k[0] + k[1]) * len(v)
#     return batches
#
#
# def load_data(data_path_1, data_path_2, max_seq_length):
#     """
#     Loads the pickle files created in preprocess-data.py
#             Parameters:
#                         data_path_1 (str): Path to the English pickle file processed in process-data.py
#                         data_path_2 (str): Path to the French pickle file processed in process-data.py
#                         max_seq_length (int): Maximum number of tokens in each sentence pair
#
#             Returns:
#                     data_1 (arr): Array of tokenized English sentences
#                     data_2 (arr): Array of tokenized French sentences
#                     data_lengths (dict): A dict with keys of tuples (length of English sentence, length of corresponding French sentence)
#                                          and values of the indices that correspond to these parallel sentences
#     """
#     with open(data_path_1, 'rb') as f:
#         data_1 = pickle.load(f)
#     with open(data_path_2, 'rb') as f:
#         data_2 = pickle.load(f)
#
#     data_lengths = {}
#     for i, (str_1, str_2) in enumerate(zip(data_1, data_2)):
#         if 0 < len(str_1) <= max_seq_length and 0 < len(str_2) <= max_seq_length - 2:
#             if (len(str_1), len(str_2)) in data_lengths:
#                 data_lengths[(len(str_1), len(str_2))].append(i)
#             else:
#                 data_lengths[(len(str_1), len(str_2))] = [i]
#     return data_1, data_2, data_lengths
#
#
# def getitem(idx):
#     """
#     Retrieves a batch given an index
#             Parameters:
#                         idx (int): Index of the batch
#                         data (arr): Array of tokenized sentences
#                         batches (arr): List of each batch (which consists of an array of indices)
#                         src (bool): True if the language is the source language, False if it's the target language
#
#             Returns:
#                     batch (arr): Array of tokenized English sentences, of size (num_sentences, num_tokens_in_sentence)
#                     masks (arr): key_padding_masks for the sentences, of size (num_sentences, num_tokens_in_sentence)
#     """
#
#     event = self.batches[idx]
#     if src:
#         batch = [data[i] for i in sentence_indices]
#     else:
#         # If it's in the target language, add [SOS] and [EOS] tokens
#         batch = [[2] + data[i] + [3] for i in sentence_indices]
#
#     # Get the maximum sentence length
#     seq_length = 0
#     for sentence in batch:
#         if len(sentence) > seq_length:
#             seq_length = len(sentence)
#
#     masks = []
#     for i, sentence in enumerate(batch):
#         # Generate the masks for each sentence, False if there's a token, True if there's padding
#         masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
#         # Add 0 padding
#         batch[i] = sentence + [0 for _ in range(seq_length - len(sentence))]
#
#     return np.array(batch), np.array(masks)
#
# #
#
def walk(folder_name):
    files = []
    for p, d, f in os.walk(folder_name):
        for file_name in f:

            if file_name[-5:] == 'event':
                files.append(os.path.join(p, file_name))
    return files



# # keydata = json.load(open(tension_folder + '/files_result.json','r'))
#
#
def cal_separate_file(files,i):
    return_list = []
    print(f'file name {files[i]}')
    # file_events = np.array(pickle.load(open('/home/ruiguo/dataset/lmd/lmd_more_event/R/R/T/TRRRTLE12903CA241F/e88a04b4b6e986efac223636a14d63bb_event', 'rb')))
    file_events = np.array(pickle.load(open(files[i], 'rb')))

    r = re.compile('i_\d')
    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)


    # file_events = pickle.load(open('/home/ruiguo/dataset/chinese_event/今生今世_event', 'rb'))
    # changed_file_events = file_events[1:3+num_of_tracks]
    # changed_file_events.extend(['bar'])
    # changed_file_events.extend( file_events[3+num_of_tracks:])
    # changed_file_events = np.array(changed_file_events)
    # file_events = np.array(pickle.load(open(files[i], 'rb')))


    if num_of_tracks < 1:
        print(f'omit file {files[i]} with no track')
        # return None

    header_events = file_events[:2+num_of_tracks]

    # time_signature = file_events[1]
    # tempo = file_events[2]


    bar_pos = np.where(file_events == 'bar')[0]

    bar_beginning_pos = bar_pos[::8]

    # meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
    # meta_without_track_control = np.concatenate([meta_events[0:3],np.array(track_program)],axis=0)
    # # < 16 bar
    for pos in range(len(bar_beginning_pos) - 1):
        if pos == len(bar_beginning_pos) - 2:

            # detect empty_event(
            return_events = add_control_event(file_events[bar_beginning_pos[pos]:], header_events)
        else:
            return_events = add_control_event(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],
                                              header_events)
        if return_events is not None:
            return_list.append(return_events.tolist())
        else:
            print(f'skip file {i} bar pos {pos}')
    # else:
    #     return_list = add_control_event(file_events[bar_beginning_pos[0]:], header_events)
    #     pickle.dump(return_list, open('/Users/ruiguo/Documents/mm2021/added_event','wb'))
    return return_list


def cal_separate_event(event):


    file_events = np.array(event)
    r = re.compile('i_\d')
    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)
    if num_of_tracks < 2:
        print(f'omit file {files[i]} with only one track')
        return None

    time_signature = file_events[1]
    tempo = file_events[2]


    bar_pos = np.where(file_events == 'bar')[0]

    bar_beginning_pos = bar_pos[::8]

    # meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
    # meta_without_track_control = np.concatenate([meta_events[0:3],np.array(track_program)],axis=0)
    # # < 16 bar
    for pos in range(len(bar_beginning_pos) - 1):
        if pos == len(bar_beginning_pos) - 2:

            # detect empty_event(
            return_events = add_control_event(file_events[bar_beginning_pos[pos]:], time_signature, tempo,
                                              track_program)
        else:
            return_events = add_control_event(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]],
                                              time_signature, tempo, track_program)
        if return_events is not None:
            return return_events
        else:
            print(f'skip file')


def gen_batches(files,max_token_length=2200, batch_window_size=8):


    print(f'total files {len(files)}')


    return_events = Parallel(n_jobs=1)(delayed(cal_separate_file)(files,i) for i in range(0,len(files)))
    batches = []
    for file_events in return_events:
        for event in file_events:
            batches.append(event)

    batches.sort(key=len)
    i = 0
    while i < len(batches) - 1:
        if batches[i] == batches[i + 1]:
            del batches[i + 1]
        else:
            i += 1

    batches_new = []
    this_batch_total_length = 0

    while len(batches) > 0:
        if this_batch_total_length + len(batches[0]) < max_token_length:
            if len(batches_new) > 0:
                batches_new[-1].append(batches[0])
            else:
                batches_new.append([batches[0]])
            this_batch_total_length += len(batches[0])
        else:
            if len(batches[0]) > max_token_length:
                print(
                    f'the event size {len(batches[0])} is greater than {max_token_length}, skip this file, or increase the max token length')
                this_batch_total_length = 0
            else:
                batches_new.append([batches[0]])
                this_batch_total_length = len(batches[0])
        del batches[0]
    del batches
    gc.collect()
    batch_lengths = {}
    for index, item in enumerate(batches_new):
        if len(item) not in batch_lengths:
            batch_lengths[len(item)] = [index]
        else:
            batch_lengths[len(item)].append(index)
    return batches_new, batch_lengths


def validate_event_data(batches):
    for batch in batches:
        for events in batch:
            print(f'{len(np.where(np.array(events) == "bar")[0])}')
            midi = event_2midi(events)[0]
            midi.write('./temp.mid')
            new_events = midi_2event('./temp.mid')[0]
            print(f'{len(np.where(np.array(new_events) == "bar")[0])}')
            added_control_event = cal_separate_event(new_events)
            print(f'{len(np.where(np.array(added_control_event) == "bar")[0])}')
            # for i,event in enumerate(events):
            if len(added_control_event) < len(events):
                print(f'added event length{len(added_control_event)} is less than { len(events)}')
                # else:
                #     if event != added_control_event[i]:
                #         print('not equal')


# #
# #


# def gen_new_batches(max_token_length=2400, batch_window_size=8):
#
#     batches = pickle.load(open('./sync/new_data','rb'))
#     for i,batch in enumerate(batches):
#         batches[i] = batch.tolist()
#
#     batches.sort(key=len)
#     i = 0
#     while i < len(batches) - 1:
#         if batches[i] == batches[i + 1]:
#             del batches[i + 1]
#         else:
#             i += 1
#
#     batches_new = []
#     this_batch_total_length = 0
#
#     while len(batches) > 0:
#         if this_batch_total_length + len(batches[0]) < max_token_length:
#             if len(batches_new) > 0:
#                 batches_new[-1].append(batches[0])
#             else:
#                 batches_new.append([batches[0]])
#             this_batch_total_length += len(batches[0])
#         else:
#             if len(batches[0]) > max_token_length:
#                 print(
#                     f'the event size {len(batches[0])} is greater than {max_token_length}, skip this file, or increase the max token length')
#                 this_batch_total_length = 0
#             else:
#                 batches_new.append([batches[0]])
#                 this_batch_total_length = len(batches[0])
#         del batches[0]
#     del batches
#     gc.collect()
#     batch_lengths = {}
#     for index, item in enumerate(batches_new):
#         if len(item) not in batch_lengths:
#             batch_lengths[len(item)] = [index]
#         else:
#             batch_lengths[len(item)].append(index)
#     return batches_new, batch_lengths
# #
# #
# all_batches,batch_length = gen_new_batches()
# pickle.dump(all_batches, open('./sync/all_batches_new','wb'))
# pickle.dump(batch_length, open('./sync/batch_length_new','wb'))
# sys.exit()

#
# vocab = WordVocab(all_tokens)
# event_folder = '/home/ruiguo/dataset/lmd/lmd_melody_bass_event/'
# # # # event_folder = '/home/ruiguo/dataset/valid_midi_out'
# # # # event_folder = '/home/ruiguo/dataset/chinese_event/'
# event_folder = '/home/ruiguo/dataset/lmd/lmd_more_event/'
# #
# window_size = 8
# #
# files = walk(event_folder)

# all_batches,batch_length = gen_batches(files)
# pickle.dump(all_batches, open(f'./sync/with_melody_batch','wb'))
# pickle.dump(batch_length, open(f'./sync/with_melody_batch_length','wb'))

#
# all_batches = pickle.load(open('/home/ruiguo/score_transformer/sync/with_melody_batch', 'rb'))
# batch_length = pickle.load(open('/home/ruiguo/score_transformer/sync/with_melody_batch_length', 'rb'))
#
# original_data_length = len(all_batches)
# test_ratio = 0.1
# valid_ratio = 0.1
# train_ratio = 0.8
#
# print(f'train_ratio is {train_ratio}')
# print(f'valid_ratio is {valid_ratio}')
# print(f'test_ratio is {test_ratio}')
#
#
# test_data_index = np.random.choice(len(all_batches), int(original_data_length * test_ratio), replace=False)
#
# test_data_index = np.sort(test_data_index)
#
# test_batches = np.array(all_batches)[test_data_index].tolist()
# test_batch_lengths = {}
#
# for index, item in enumerate(test_batches):
#     if len(item) not in test_batch_lengths:
#         test_batch_lengths[len(item)] = [index]
#     else:
#         test_batch_lengths[len(item)].append(index)
#
# for index in test_data_index[::-1]:
#     del all_batches[index]
#
# valid_data_index = np.random.choice(len(all_batches), int(original_data_length * valid_ratio), replace=False)
# valid_data_index = np.sort(valid_data_index)
#
# valid_batches = np.array(all_batches)[valid_data_index].tolist()
# valid_batch_lengths = {}
#
# for index, item in enumerate(valid_batches):
#     if len(item) not in valid_batch_lengths:
#         valid_batch_lengths[len(item)] = [index]
#     else:
#         valid_batch_lengths[len(item)].append(index)
#
# for index in valid_data_index[::-1]:
#     del all_batches[index]
#
#
# train_batches = all_batches
# train_batch_lengths = {}
#
# for index, item in enumerate(train_batches):
#     if len(item) not in train_batch_lengths:
#         train_batch_lengths[len(item)] = [index]
#     else:
#         train_batch_lengths[len(item)].append(index)
#
#
# print(f'train batch length is {len(train_batches)}')
# print(f'valid batch length is {len(valid_batches)}')
# print(f'valid batch length is {len(test_batches)}')
#
# pickle.dump(train_batches, open('/home/data/guorui/score_transformer/melody_train_batches','wb'))
# pickle.dump(valid_batches, open('/home/data/guorui/score_transformer/melody_valid_batches','wb'))
# pickle.dump(test_batches, open('/home/data/guorui/score_transformer/melody_test_batches','wb'))
#
# pickle.dump(train_batch_lengths, open('/home/data/guorui/score_transformer/melody_train_batch_lengths','wb'))
# pickle.dump(valid_batch_lengths, open('/home/data/guorui/score_transformer/melody_valid_batch_lengths','wb'))
# pickle.dump(test_batch_lengths, open('/home/data/guorui/score_transformer/melody_test_batch_lengths','wb'))
# sys.exit()



#     pickle.dump(batch_length, open(f'./sync/batch_length_{j}','wb'))
# for i in range(0,45000,15000):
#     j = str(int(i/15000))
#     all_batches,batch_length = gen_batches(files[i:i+15000])
#     pickle.dump(all_batches, open(f'./sync/all_batches_{j}','wb'))
#     pickle.dump(batch_length, open(f'./sync/batch_length_{j}','wb'))
# all_batches,batch_length = gen_batches(files[45000:])
# j = str(int(45000/15000))
# pickle.dump(all_batches, open(f'./sync/all_batches_{j}','wb'))
# pickle.dump(batch_length, open(f'./sync/batch_length_{j}','wb'))
#
#
# sys.exit()

# validate a few event data
#
# all_batches = pickle.load(open('/home/ruiguo/score_transformer/sync/all_batches_3', 'rb'))
# batch_length = pickle.load(open('/home/ruiguo/score_transformer/sync/batch_length_3', 'rb'))
#
# original_data_length = len(all_batches)
# train_4_ratio = 0.5
#
# train_4_data_index = np.random.choice(len(all_batches), int(original_data_length * train_4_ratio), replace=False)
#
# train_4_data_index = np.sort(train_4_data_index)
#
# train_4_batches = np.array(all_batches)[train_4_data_index].tolist()
# train_4_batch_lengths = {}
#
# for index, item in enumerate(train_4_batches):
#     if len(item) not in train_4_batch_lengths:
#         train_4_batch_lengths[len(item)] = [index]
#     else:
#         train_4_batch_lengths[len(item)].append(index)
#
# for index in train_4_data_index[::-1]:
#     del all_batches[index]
#
#
# valid_data_index = np.random.choice(len(all_batches), int(len(all_batches) * 0.5), replace=False)
#
# valid_data_index = np.sort(valid_data_index)
#
# valid_batches = np.array(all_batches)[valid_data_index].tolist()
# valid_batch_lengths = {}
#
#
# for index, item in enumerate(valid_batches):
#     if len(item) not in valid_batch_lengths:
#         valid_batch_lengths[len(item)] = [index]
#     else:
#         valid_batch_lengths[len(item)].append(index)
#
#
# for index in valid_data_index[::-1]:
#     del all_batches[index]
#
# test_batches = np.array(all_batches).tolist()
# test_batch_lengths = {}
#
# for index, item in enumerate(test_batches):
#     if len(item) not in test_batch_lengths:
#         test_batch_lengths[len(item)] = [index]
#     else:
#         test_batch_lengths[len(item)].append(index)
#
#
# print(f'valid batch length is {len(valid_batches)}')
# print(f'test batch length is {len(test_batches)}')
#
#
# pickle.dump(train_4_batches, open('./sync/all_batches_4','wb'))
# pickle.dump(train_4_batch_lengths, open('./sync/batch_length_4','wb'))
# pickle.dump(valid_batches, open('./sync/valid_batches_new','wb'))
# pickle.dump(test_batches, open('./sync/test_batches_new','wb'))
# pickle.dump(valid_batch_lengths, open('./sync/valid_batch_lengths_new','wb'))
# pickle.dump(test_batch_lengths, open('./sync/test_batch_lengths_new','wb'))
# #
# sys.exit()





#
# validate_event_data(all_batches)


# #

#
#
# all_batches = pickle.load(open('/home/ruiguo/score_transformer/sync/all_batches', 'rb'))
# batch_length = pickle.load(open('/home/ruiguo/score_transformer/sync/batch_length', 'rb'))
# #
# original_data_length = len(all_batches)
# test_ratio = 0.1
# valid_ratio = 0.1
# train_ratio = 0.5
# # three separate training data for generating new data in training
# separate_training_data_ratio = 0.1
#
# print(f'train_ratio is {train_ratio}')
# print(f'valid_ratio is {valid_ratio}')
# print(f'test_ratio is {test_ratio}')
#
#
#
# test_data_index = np.random.choice(len(all_batches), int(original_data_length * test_ratio), replace=False)
#
# test_data_index = np.sort(test_data_index)
#
# test_batches = np.array(all_batches)[test_data_index].tolist()
# test_batch_lengths = {}
#
# for index, item in enumerate(test_batches):
#     if len(item) not in test_batch_lengths:
#         test_batch_lengths[len(item)] = [index]
#     else:
#         test_batch_lengths[len(item)].append(index)
#
# for index in test_data_index[::-1]:
#     del all_batches[index]
#
# valid_data_index = np.random.choice(len(all_batches), int(original_data_length * valid_ratio), replace=False)
# valid_data_index = np.sort(valid_data_index)
#
# valid_batches = np.array(all_batches)[valid_data_index].tolist()
# valid_batch_lengths = {}
#
# for index, item in enumerate(valid_batches):
#     if len(item) not in valid_batch_lengths:
#         valid_batch_lengths[len(item)] = [index]
#     else:
#         valid_batch_lengths[len(item)].append(index)
#
# for index in valid_data_index[::-1]:
#     del all_batches[index]
#
# for i in range(3):
#     separate_training_data_index = np.random.choice(len(all_batches), int(original_data_length * separate_training_data_ratio), replace=False)
#     separate_training_data_index = np.sort(separate_training_data_index)
#
#     separate_training_batches = np.array(all_batches)[separate_training_data_index].tolist()
#     separate_training_batch_lengths = {}
#
#     for index, item in enumerate(separate_training_batches):
#         if len(item) not in separate_training_batch_lengths:
#             separate_training_batch_lengths[len(item)] = [index]
#         else:
#             separate_training_batch_lengths[len(item)].append(index)
#     pickle.dump(separate_training_batches, open(f'./sync/separate_training_batches_{i}','wb'))
#     pickle.dump(separate_training_batch_lengths, open(f'./sync/separate_training_batch_lengths_{i}', 'wb'))
#
#     for index in separate_training_data_index[::-1]:
#         del all_batches[index]
#
# train_batches = all_batches
# train_batch_lengths = {}
#
# for index, item in enumerate(train_batches):
#     if len(item) not in train_batch_lengths:
#         train_batch_lengths[len(item)] = [index]
#     else:
#         train_batch_lengths[len(item)].append(index)
#
#
# print(f'train batch length is {len(train_batches)}')
# print(f'valid batch length is {len(valid_batches)}')
# print(f'valid batch length is {len(test_batches)}')
#
# print(f'separate training batch length is {len(separate_training_batches)}')
# pickle.dump(train_batches, open('./sync/train_batches','wb'))
# pickle.dump(valid_batches, open('./sync/valid_batches','wb'))
# pickle.dump(test_batches, open('./sync/test_batches','wb'))
#
# pickle.dump(train_batch_lengths, open('./sync/train_batch_lengths','wb'))
# pickle.dump(valid_batch_lengths, open('./sync/valid_batch_lengths','wb'))
# pickle.dump(test_batch_lengths, open('./sync/test_batch_lengths','wb'))
# sys.exit()

# folder_prefix = '/home/ruiguo/'
# test_batches = pickle.load(open(folder_prefix + 'score_transformer/sync/test_batches_0_0_8_new_bins', 'rb'))
# test_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/sync/test_batch_lengths_0_0_8_new_bins', 'rb'))
#
# #
# span_ratio_separately_each_epoch = np.array([[1, 0, 0], [.5, .5, 0],
#                                              [.25, .75, 0], [.25, .5, .25],
#                                              [.25, .25, .5]])
#
#
# test_dataset = ParallelLanguageDataset('', '',
#                                            vocab, 0,
#                                            0,
#                                            2200,
#                                            16,
#                                            test_batches,
#                                            test_batch_lengths,
#                                            .15,
#                                            .3,
#                                            .3,
#                                            .3,
#                                            0,
#                                            .3,
#                                            3,
#                                            0.5,
#                                            span_ratio_separately_each_epoch,
#                                            True)
# data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=lambda batch: collate_mlm(batch))  # 训练语料按长度排好序的
#
# for i in data_loader:
#     print(i)
# files = ['/Users/ruiguo/Documents/mm2021/temp_event']
# cal_separate_file(files,0)
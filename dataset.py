import random
import torch
import gc
from torch.utils.data import Dataset
from preprocessing import event_2midi
from einops import rearrange
import re
import json
import os
from vocab import *
import sys

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
                 span_lengths,
                 span_ratio_jointly,
                 span_ratio_separately_each_epoch,
                 mask_bar_num_ratio,
                 mask_track_num_ratio,
                 mask_bar_ctrl_token=False,
                 pretraining=True,
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
        self.span_lengths = span_lengths
        self.span_ratio_jointly = span_ratio_jointly
        self.span_ratio_separately_each_epoch = span_ratio_separately_each_epoch
        self.train_jointly = train_jointly
        self.epoch = 0
        self.previous_index = 0
        self.mask_bar_num_ratio = mask_bar_num_ratio,
        self.mask_track_num_ratio = mask_track_num_ratio,
        self.mask_bar_ctrl_token = mask_bar_ctrl_token,
        self.train_jointly = train_jointly
        self.pretraining = pretraining

        print(f'pretraining is {self.pretraining}')
        if self.pretraining:
            print(f'control mask ratio is {self.control_mask_ratio}')
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

        length = len(self.batches[this_idx])
        return_idx = random.choice(self.batch_lengths[length])

        event = self.batches[return_idx]

        if self.pretraining:
            masked_input, decoder_in, decoder_target = self.random_word(event,
                                                                        self.total_mask_ratio,
                                                                        self.span_lengths,
                                                                        self.span_ratio_jointly,
                                                                        self.span_ratio_separately_each_epoch,
                                                                        self.epoch,
                                                                        self.train_jointly)
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

            events_with_control,keys = add_control_event(file_events, bar_pos, tensiles, diameters,keys)

            bar_pos = np.where(events_with_control == 'bar')[0]
            # total_bars = min(len(tensiles), len(diameters), len(bar_pos))
            # bar_pos = bar_pos[:total_bars]

            bar_beginning_pos = bar_pos[::batch_window_size]

            meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0]-2]

            for pos in range(len(bar_beginning_pos) - 1):

                # print(bar_beginning_pos[pos])
                if keys[2] != -1 and pos*8+1 >= keys[2]:
                    meta_events[2] = key_to_token[keys[3]]
                if pos == len(bar_beginning_pos) - 2:
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
            if self.epoch < -1:
                while total_masked_ratio < total_ratio and start_pos < len(event):
                    masked_token = []
                    prob = random.random()


                    if prob < random_threshold:
                        prob /= random_threshold

                        if prob < span_ratio[0]:
                            if start_pos + span_lengths[0] <= len(event):
                                masked_token = event[start_pos:start_pos + span_lengths[0]]
                                tokens.append(self.vocab.mask_indices[masked_num])
                                total_masked_ratio += span_lengths[0] / len(event)
                                start_pos += span_lengths[0]
                        elif span_ratio[0] < prob < span_ratio[1] + span_ratio[0]:
                            if start_pos + span_lengths[1] <= len(event):
                                masked_token = event[start_pos:start_pos + span_lengths[1]]
                                tokens.append(self.vocab.mask_indices[masked_num])
                                total_masked_ratio += span_lengths[1] / len(event)
                                start_pos += span_lengths[1]
                        else:
                            if start_pos + span_lengths[2] <= len(event):
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
            else:
                for token in event:
                    if self.vocab.char2index(token) in self.vocab.control_indices:
                        total_control_tokens += 1

                while total_masked_ratio < total_ratio and start_pos < len(event) and control_masked_ratio['total'] < self.control_mask_ratio:
                    masked_token = []
                    prob = random.random()
                    have_control_token = False
                    control_token_length = 0
                    if prob < span_ratio[0]:
                        if start_pos + span_lengths[0] <= len(event):
                            for event_token in event[start_pos:start_pos + span_lengths[0]]:
                                if self.vocab.char2index(event_token) in self.vocab.control_indices:
                                    have_control_token = True
                                    control_token_length += 1

                            if have_control_token:
                                prob = random.random()
                                if prob < self.control_mask_ratio:
                                    masked_token = event[start_pos:start_pos + span_lengths[0]]
                                    tokens.append(self.vocab.mask_indices[masked_num])
                                    total_masked_ratio += span_lengths[0] / len(event)
                                    control_masked_ratio['total'] += control_token_length / total_control_tokens
                                    start_pos += span_lengths[0]
                            else:
                                prob = random.random()
                                if prob < random_threshold * 1.5:
                                    masked_token = event[start_pos:start_pos + span_lengths[0]]
                                    tokens.append(self.vocab.mask_indices[masked_num])
                                    total_masked_ratio += span_lengths[0] / len(event)
                                    start_pos += span_lengths[0]

                    elif span_ratio[0] < prob < span_ratio[1] + span_ratio[0]:
                        if start_pos + span_lengths[1] <= len(event):
                            for event_token in event[start_pos:start_pos + span_lengths[1]]:
                                if self.vocab.char2index(event_token) in self.vocab.control_indices:
                                    have_control_token = True
                                    control_token_length += 1

                            if have_control_token:
                                prob = random.random()
                                if prob < self.control_mask_ratio:
                                    masked_token = event[start_pos:start_pos + span_lengths[1]]
                                    tokens.append(self.vocab.mask_indices[masked_num])
                                    control_masked_ratio['total'] += control_token_length / total_control_tokens
                                    total_masked_ratio += span_lengths[1] / len(event)
                                    start_pos += span_lengths[1]
                            else:
                                prob = random.random()
                                if prob < random_threshold * 1.5:
                                    masked_token = event[start_pos:start_pos + span_lengths[1]]
                                    tokens.append(self.vocab.mask_indices[masked_num])
                                    total_masked_ratio += span_lengths[1] / len(event)
                                    start_pos += span_lengths[1]
                    else:
                        if start_pos + span_lengths[2] <= len(event):
                            for event_token in event[start_pos:start_pos + span_lengths[2]]:
                                if self.vocab.char2index(event_token) in self.vocab.control_indices:
                                    have_control_token = True
                                    control_token_length += 1

                            if have_control_token:
                                prob = random.random()
                                if prob < self.control_mask_ratio:
                                    masked_token = event[start_pos:start_pos + span_lengths[2]]
                                    tokens.append(self.vocab.mask_indices[masked_num])
                                    control_masked_ratio['total'] += control_token_length / total_control_tokens
                                    total_masked_ratio += span_lengths[2] / len(event)
                                    start_pos += span_lengths[2]
                            else:
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

            # current_track_unmasked = True
            #
            # current_track = 0
            # unmasked = True

            # while total_masked_ratio < self.total_mask_ratio and start_pos < len(event):
            #
            #     prob = random.random()
            #
            #     # add track selection
            #     # if event[start_pos] == 'track_0':
            #     #     current_track = 0
            #     #     current_track_unmasked = True
            #     # if event[start_pos] == 'track_1':
            #     #     current_track = 1
            #     #     current_track_unmasked = True
            #     # if event[start_pos] == 'track_2':
            #     #     current_track = 2
            #     #     current_track_unmasked = True
            #
            #
            #     for i, token_type in enumerate(token_types):
            #         if event[start_pos] in token_type:
            #             # current_track_unmasked and current_track == 0
            #             if prob < ratios[i]:
            #
            #                 masked_token = event[start_pos]
            #                 tokens.append(self.vocab.mask_indices[masked_num])
            #                 total_masked_ratio += 1 / len(event)
            #                 start_pos += 1
            #
            #                 if len(masked_token) > 0:
            #                     if not isinstance(masked_token, list):
            #                         masked_token = [masked_token]
            #                     decoder_in.append(self.vocab.mask_indices[masked_num])
            #                     for token in masked_token:
            #                         decoder_in.append(self.vocab.char2index(token))
            #                         decoder_target.append(self.vocab.char2index(token))
            #                     else:
            #                         decoder_target.append(self.vocab.eos_index)
            #
            #
            #                     # current_track_unmasked = False
            #                     # masked_num += 1
            #
            #                     # unmasked  = False
            #                     break
            #
            #             else:
            #                 tokens.append(self.vocab.char2index(event[start_pos]))
            #                 start_pos += 1
            #                 break

            while start_pos < len(event):
                tokens.append(self.vocab.char2index(event[start_pos]))
                start_pos += 1

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

    def mask_category(self,events, token_type, number):

        total_tokens = []
        total_decoder_in = []
        total_decoder_target = []

        random.shuffle(events)
        for event in events:
            tokens = []
            decoder_in = []
            decoder_target = []
            start_pos = 0
            total_masked_ratio = 0
            masked_num = 0
            in_event = False
            previous_pos = 0
            while total_masked_ratio < self.total_mask_ratio and start_pos < len(event):


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

                if self.vocab.token_class_ranges[self.vocab.char2index(event[start_pos])] == token_type:

                    if masked_num < number:
                        if in_event is False:
                            masked_token = [event[start_pos]]
                            tokens.append(self.vocab.mask_indices[0])
                            in_event = True
                            prevous_pos = start_pos
                        else:
                            masked_token.append(event[start_pos])

                        masked_num += 1
                        total_masked_ratio += 1 / len(event)
                        start_pos += 1


                    else:
                        if in_event:
                            decoder_in.append(self.vocab.mask_indices[0])
                            for token in masked_token:
                                decoder_in.append(self.vocab.char2index(token))
                                decoder_target.append(self.vocab.char2index(token))
                            else:
                                decoder_target.append(self.vocab.eos_index)
                            in_event = False
                        tokens.append(self.vocab.char2index(event[start_pos]))
                        start_pos += 1



                else:
                    if in_event:
                        decoder_in.append(self.vocab.mask_indices[0])
                        for token in masked_token:
                            decoder_in.append(self.vocab.char2index(token))
                            decoder_target.append(self.vocab.char2index(token))
                        else:
                            decoder_target.append(self.vocab.eos_index)
                        in_event = False
                    tokens.append(self.vocab.char2index(event[start_pos]))
                    start_pos += 1

            while start_pos < len(event):
                tokens.append(self.vocab.char2index(event[start_pos]))
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


    def mask_bars(self,events,bar_ratio,track_ratio,mask_bar_ctr=False):

        # mask bar token (w/wo bar control token) and try to generate bar token

        total_tokens = []
        total_decoder_in = []
        total_decoder_target = []

        random.shuffle(events)
        for event in events:
            tokens = []
            decoder_in = []
            decoder_target = []
            masked_indices_pairs = []

            bar_pos = np.where(np.array(event) == 'bar')[0]

            r = re.compile('i_\d')

            track_program = list(filter(r.match, event))
            track_num = len(track_program)


            if track_num == 3:
                ratios = track_ratio[0]
                track_0_pos = np.where('track_0' == np.array(event))[0]
                track_1_pos = np.where('track_1' == np.array(event))[0]
                track_2_pos = np.where('track_2' == np.array(event))[0]
                all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos, track_2_pos]))

            else:
                ratios = track_ratio[1]
                track_0_pos = np.where('track_0' == np.array(event))[0]
                track_1_pos = np.where('track_1' == np.array(event))[0]
                all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos]))
            bar_number_prob = np.array(bar_ratio[:len(bar_pos)],dtype=float)
            if not np.any(bar_number_prob):
                bar_number_prob = np.ones(len(bar_pos))

            print(f'bar number prob is {bar_number_prob}')
            bar_number_prob /= np.sum(bar_number_prob)

            # if np.sum(bar_number_prob) != 1:
            #     print(bar_number_prob)
            # print(f'bar number prob is {bar_number_prob}')
            resample_counts = np.random.multinomial(1, bar_number_prob[:len(bar_pos)])
            bar_mask_number = np.where(resample_counts)[0][0] + 1
            # bar_mask_number = np.random.choice(range(len(bar_pos)), size=1, p=bar_ratio[:len(bar_pos)])
            bar_start_indices = np.sort(np.random.choice(len(bar_pos),size=bar_mask_number,replace=False))
            print(f'bar start indices is {bar_start_indices}')
            for bar_start_index in bar_start_indices:
                # bar_start_index = np.random.choice(len(bar_pos))


                if bar_start_index == len(bar_pos) - 1:
                    next_bar_start_pos = len(event)
                else:
                    next_bar_start_index = bar_start_index + 1
                    next_bar_start_pos = bar_pos[next_bar_start_index]

                bar_start_pos = bar_pos[bar_start_index]


                track_start_index = np.where(all_track_pos > bar_start_pos)[0][0]
                track_positions = all_track_pos[track_start_index:track_start_index + track_num]

                track_positions = np.append(track_positions, next_bar_start_pos)

                prob = random.random()
                # print(prob)
                if prob < ratios[0]:
                    # select one track
                    track_pos_select_index = [np.random.choice(track_num)]
                elif prob < ratios[0] + ratios[1]:
                    # select two tracks
                    track_pos_select_index = np.sort(np.random.choice(track_num, 2, replace=False))
                else:
                    # select three tracks
                    track_pos_select_index = np.arange(track_num)

                if self.verbose:
                    print(f'mask bar {bar_start_index + 1} track {track_pos_select_index}')

                for track_pos_index in track_pos_select_index:

                    track_start_pos = track_positions[track_pos_index]
                    if track_pos_index + 1 == len(track_positions):
                        print('why')
                    track_end_pos = track_positions[track_pos_index + 1]
                    # print(track_start_pos)
                    # print(track_end_pos)
                    masked_indices_pairs.append((track_start_pos,track_end_pos))
                    # track_event = event[track_start_pos:track_end_pos]
                    # print(track_event)

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
                # print(f'event length is {len(event)}')
                # print(f'tokens length is {len(tokens)}')
                # print(f'masked num is {masked_num}')
                # print(f'decoder_in length is {len(decoder_in)}')
                # print(f'decoder_out length is {len(decoder_target)}')
                # print(f'ratio is {(len(tokens) + len(decoder_in)) / len(event)}')
                total_tokens.append(tokens)
                total_decoder_in.append(decoder_in)
                total_decoder_target.append(decoder_target)

        if len(total_tokens) == 0:
            print('why')





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
            registers.append(int(np.mean(register) / 21))
    return registers


#


def add_control_event(file_events, tensiles=None, diameters=None, keys=None):
    file_events = np.copy(file_events)
    r = re.compile('i_\d')

    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)

    bar_pos = np.where(file_events == 'bar')[0]

    if tensiles is not None:
        total_bars = min(len(tensiles), len(diameters), len(bar_pos))
        if total_bars < len(bar_pos):
            bar_pos = bar_pos[:total_bars + 1]
            file_events = file_events[:bar_pos[-1]]
            bar_pos = bar_pos[:-1]


    #     print(f'number of bars is {len(bar_pos)}')
    #     print(f'time signature is {file_event[1]}')
    bar_length = int(file_events[1][0])

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
        for track_index in range(len(track_names) - 1):
            track_event = bar_events[track_pos[track_index]:track_pos[track_index + 1]]
            track_events[track_names[track_index]].append(track_event)
        #             print(track_event)
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

    if keys:
        key = key_to_token[keys[0]]
        file_events.insert(3, key)


    for token in track_control_tokens[::-1]:
        file_events.insert(4, token)

    if '_' not in file_events[2]:
        tempo = float(file_events[2])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        file_events[2] = f't_{tempo_category}'

    if tensiles is not None:
        tensile_category = to_category(tensiles, tensile_bins)
        diameter_category = to_category(diameters, diameter_bins)

        tension_positions = np.where(np.array(file_events) == 'track_0')[0]

        total_insert = 0

        for i, pos in enumerate(tension_positions):
            file_events.insert(pos + total_insert, f's_{tensile_category[i]}')
            total_insert += 1
            file_events.insert(pos + total_insert, f'a_{diameter_category[i]}')
            total_insert += 1
    if keys:
        return np.array(file_events), keys
    else:
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

def walk(folder_name):
    files = []
    for p, d, f in os.walk(folder_name):
        for file_name in f:

            if file_name[-5:] == 'event':
                files.append(os.path.join(p, file_name))
    return files
vocab = WordVocab(all_tokens)
event_folder = '/home/ruiguo/dataset/lmd/lmd_separate_event/'
tension_folder = '/home/ruiguo/dataset/lmd/lmd_tension_three_tracks'
file_size = 100
window_size = 8

files = walk(event_folder)
#
keydata = json.load(open(tension_folder + '/files_result.json','r'))
#
def gen_batches(files, key_data, max_token_length=2400, batch_window_size=8):
    batches = []
    for i in range(len(files)):

        file_events = np.array(pickle.load(open(files[i], 'rb')))
        num_of_tracks = len(file_events[3:np.where('track_0' == file_events)[0][0]])
        if num_of_tracks < 2:
            print(f'omit file {files[i]} with only one track')
            continue

        file_name_in_folder = files[i].split('lmd_separate_event')[1:][0][:-6]
        tensile_file = tension_folder + file_name_in_folder + '.tensile'
        diameter_file = tension_folder + file_name_in_folder + '.diameter'

        tensiles = np.array(pickle.load(open(tensile_file, 'rb')))
        diameters = np.array(pickle.load(open(diameter_file, 'rb')))
        if tension_folder + file_name_in_folder + '.mid' in key_data:
            keys = key_data[tension_folder + file_name_in_folder + '.mid']
        else:
            print(f'omit file {files[i]} with no key')
            continue
        # if keys[2] != -1:
        #     print(f'file name is {files[i]}')

        events_with_control, keys = add_control_event(file_events, tensiles, diameters, keys)

        bar_pos = np.where(events_with_control == 'bar')[0]
        # total_bars = min(len(tensiles), len(diameters), len(bar_pos))
        # bar_pos = bar_pos[:total_bars]

        bar_beginning_pos = bar_pos[::batch_window_size]

        meta_events = events_with_control[1:np.where('track_0' == events_with_control)[0][0] - 2]
        meta_without_track_control = np.concatenate([meta_events[0:3],meta_events[-3:]],axis=0)
        if len(bar_beginning_pos) <= 2:
            return_events = events_with_control
            r = re.compile('i_\d')

            if len(list(filter(r.match, return_events.tolist()))) > 3:
                print('invalid')

            batches.append(return_events.tolist())
        else:
            for pos in range(len(bar_beginning_pos) - 1):

                # print(bar_beginning_pos[pos])
                if keys[2] != -1 and pos * 8 + 1 >= keys[2]:
                    meta_without_track_control[2] = key_to_token[keys[3]]
                if pos == len(bar_beginning_pos) - 2:
                    # skip the last one
                    # continue
                    # return_events = file_events[bar_beginning_pos[pos]:]
                    events_with_header = np.insert(events_with_control[bar_beginning_pos[pos]:], 1, meta_without_track_control)
                    return_events = add_control_event(events_with_header)

                elif pos > 0:

                    events_with_header = np.insert(events_with_control[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]], 1,
                                              meta_without_track_control)
                    return_events = add_control_event(events_with_header)
                # no need to change for first one
                else:
                    return_events = events_with_control[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]]

                r = re.compile('i_\d')

                if len(list(filter(r.match, return_events.tolist()))) > 3:
                    print('invalid')

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

#
# all_batches,batch_length = gen_batches(files,keydata)
# pickle.dump(all_batches, open('all_batches_new_bins','wb'))
# pickle.dump(batch_length, open('batch_length_new_bins','wb'))


#
# ratio_to_generate = [.1,.5,.8]
#
# for ratio in ratio_to_generate:
#     all_batches = pickle.load(open('/home/ruiguo/score_transformer/all_batches_new_bins', 'rb'))
#     batch_length = pickle.load(open('/home/ruiguo/score_transformer/batch_length_new_bins', 'rb'))
#
#     original_data_length = len(all_batches)
#     test_ratio = ratio/10
#     valid_ratio = test_ratio
#     train_ratio = ratio
#     print(f'train_ratio is {train_ratio}')
#     print(f'valid_ratio is {valid_ratio}')
#     print(f'test_ratio is {test_ratio}')
#
#     output_ratio_name = str(int(train_ratio * 10))
#
#     test_data_index = np.random.choice(len(all_batches), int(original_data_length * test_ratio), replace=False)
#
#     test_data_index = np.sort(test_data_index)
#
#     test_batches = np.array(all_batches)[test_data_index].tolist()
#     test_batch_lengths = {}
#
#     for index, item in enumerate(test_batches):
#         if len(item) not in test_batch_lengths:
#             test_batch_lengths[len(item)] = [index]
#         else:
#             test_batch_lengths[len(item)].append(index)
#
#     for index in test_data_index[::-1]:
#         del all_batches[index]
#
#     valid_data_index = np.random.choice(len(all_batches), int(original_data_length * valid_ratio), replace=False)
#     valid_data_index = np.sort(valid_data_index)
#
#     valid_batches = np.array(all_batches)[valid_data_index].tolist()
#     valid_batch_lengths = {}
#
#     for index, item in enumerate(valid_batches):
#         if len(item) not in valid_batch_lengths:
#             valid_batch_lengths[len(item)] = [index]
#         else:
#             valid_batch_lengths[len(item)].append(index)
#
#     for index in valid_data_index[::-1]:
#         del all_batches[index]
#
#     train_data_index = np.random.choice(len(all_batches), int(original_data_length * train_ratio), replace=False)
#     to_delete_index = np.setdiff1d(np.arange(len(all_batches)), train_data_index)
#
#     for index in to_delete_index[::-1]:
#         del all_batches[index]
#
#     train_batches = all_batches
#     train_batch_lengths = {}
#
#     for index, item in enumerate(train_batches):
#         if len(item) not in train_batch_lengths:
#             train_batch_lengths[len(item)] = [index]
#         else:
#             train_batch_lengths[len(item)].append(index)
#
#
#
#     print(f'train batch length is {len(train_batches)}')
#     print(f'valid batch length is {len(valid_batches)}')
#     print(f'valid batch length is {len(test_batches)}')
#     pickle.dump(train_batches, open('train_batches_0_' + output_ratio_name + '_new_bins','wb'))
#     pickle.dump(valid_batches, open('valid_batches_0_0_' + output_ratio_name + '_new_bins','wb'))
#     pickle.dump(test_batches, open('test_batches_0_0_' + output_ratio_name + '_new_bins','wb'))
#
#     pickle.dump(train_batch_lengths, open('train_batch_lengths_0_' + output_ratio_name + '_new_bins','wb'))
#     pickle.dump(valid_batch_lengths, open('valid_batch_lengths_0_0_' + output_ratio_name + '_new_bins','wb'))
#     pickle.dump(test_batch_lengths, open('test_batch_lengths_0_0_' + output_ratio_name + '_new_bins','wb'))
# sys.exit()

# folder_prefix = '/home/ruiguo/'
# test_batches = pickle.load(open(folder_prefix + 'score_transformer/test_batches_0_0_8_new_bins', 'rb'))
# test_batch_lengths = pickle.load(open(folder_prefix + 'score_transformer/test_batch_lengths_0_0_8_new_bins', 'rb'))
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

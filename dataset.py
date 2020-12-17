import pickle
import random
import numpy as np
import os
import torch
import gc

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

pad = '<pad>'
eos = '<eos>'
mask = [f'm_{num}' for num in range(1)]
special_tokens = [pad, eos]

time_signature_token = ['4/4', '3/4', '2/4', '6/8']

program_num = [f'i_{num}' for num in range(128)]

tempo_token = [f't{i}' for i in range(8)]

track_num = [f'track_{num}' for num in range(3)]

structure_token = ['bar'] + track_num

header_token = time_signature_token + tempo_token + program_num

control_tokens = []

rests = ['rest_e', 'rest_s']

durations = ['whole', 'half', 'quarter', 'eighth', 'sixteenth']

pitches = [f'p{num}' for num in range(21, 109)] + rests + ['continue']

note_tokens = pitches + durations

all_tokens = special_tokens + structure_token + \
             header_token + note_tokens + mask


class WordVocab(object):
    def __init__(self, char_lst):
        super(WordVocab, self).__init__()
        self.pad_index = 0
        self.eos_index = 1
        self.char_lst = char_lst
        self._char2idx = {
            '<pad>': self.pad_index,
            '<eos>': self.eos_index,
        }

        for char in self.char_lst:
            if char not in self._char2idx:
                self._char2idx[char] = len(self._char2idx)
        self._idx2char = dict((idx, char) for char, idx in self._char2idx.items())
        print(f'vocab size: {self.vocab_size}')

        self.token_class_ranges = {}
        self.structure_indices = [self._char2idx[name] for name in structure_token]
        self.pitch_indices = [self._char2idx[name] for name in pitches]
        self.mask_indices = [self._char2idx[name] for name in mask]
        self.duration_indices = [self._char2idx[name] for name in durations + rests]
        self.control_indices = [self._char2idx[name] for name in time_signature_token + \
                           tempo_token]
        self.program_indices = [self._char2idx[name] for name in program_num]
        for index in self.structure_indices:
            self.token_class_ranges[index] = 'structure'
        for index in self.pitch_indices:
            self.token_class_ranges[index] = 'pitch'
        for index in self.duration_indices:
            self.token_class_ranges[index] = 'duration'
        for index in self.mask_indices:
            self.token_class_ranges[index] = 'mask'
        for index in self.control_indices:
            self.token_class_ranges[index] = 'control'
        for index in self.program_indices:
            self.token_class_ranges[index] = 'program'
        self.token_class_ranges[self.eos_index] = 'eos'

    def char2index(self, token):

        return self._char2idx.get(token)

    def index2char(self, idxs):

        return self._idx2char.get(idxs)

    def get_token_classes(self, idx):
        return self.token_class_ranges[idx]

    @property
    def vocab_size(self):
        return len(self._char2idx)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


class ParallelLanguageDataset(Dataset):
    def __init__(self, event_folder, vocab, start_ratio, end_ratio, max_token_length, window_size,
                 total_mask_ratio, structure_mask_ratio,
                 duration_mask_ratio, pitch_mask_ratio,
                 control_mask_ratio, header_mask_ratio):
        """
        Initializes the dataset
                Parameters:
                        data_path_1 (str): Path to the English pickle file processed in process-data.py
                        data_path_2 (str): Path to the French pickle file processed in process-data.py
                        num_tokens (int): Maximum number of tokens in each batch (restricted by GPU memory usage)
                        max_seq_length (int): Maximum number of tokens in each sentence pair
        """
        self.event_folder = event_folder
        self.vocab = vocab
        self.files = self.walk(self.event_folder)
        start_idx = int(len(self.files) * start_ratio)
        end_idx = int(len(self.files) * end_ratio)
        print(f'start idx is {start_idx}')
        print(f'end idx is {end_idx}')
        self.files = self.files[start_idx:end_idx]

        self.batches = self.gen_batches(self.files, max_token_length, window_size)
        self.total_mask_ratio = total_mask_ratio

        self.structure_mask_ratio = structure_mask_ratio
        self.duration_mask_ratio = duration_mask_ratio
        self.pitch_mask_ratio = pitch_mask_ratio
        self.control_mask_ratio = control_mask_ratio
        self.header_mask_ratio = header_mask_ratio

    def __getitem__(self, idx):
        event = self.batches[idx]

        masked_input, decoder_in, decoder_target = self.random_word(event)

        # src, src_mask, tgt, tgt_mask = getitem(idx)
        # tgt, tgt_mask = getitem(idx)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        # mlm_input = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]  # 3，1，2
        # mlm_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]

        # return mlm_input, mlm_label

        return masked_input, decoder_in, decoder_target

    def __len__(self):
        return len(self.batches)

    def gen_batches(self, files, max_token_length=2200, batch_window_size=8):

        batches = []

        for i in range(len(files)):
            file_events = np.array(pickle.load(open(files[i], 'rb')))
            tempo = float(file_events[2])
            tempo_bins = np.array([0] + list(range(60, 190, 20)) + [1000])

            tempo_category = int(np.where((tempo - tempo_bins) > 0)[0][-1])
            file_events[2] = f't{tempo_category}'

            bar_pos = np.where(file_events == 'bar')[0]

            bar_beginning_pos = bar_pos[::batch_window_size]

            meta_events = file_events[1:np.where('track_0' == file_events)[0][0]]

            for pos in range(len(bar_beginning_pos) - 1):

                # print(bar_beginning_pos[pos])

                if pos == len(bar_beginning_pos) - 2:
                    # skip the last one
                    continue
                    return_events = file_events[bar_beginning_pos[pos]:]
                    # return_events = np.insert(file_events[bar_beginning_pos[pos]:], 1, meta_events)
                # elif pos > 0:
                #     return_events = np.insert(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]], 1,
                #                               meta_events)
                else:
                    return_events = file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 2]]
                batches.append(return_events.tolist())
        batches.sort(key=len)
        i = 0
        while i < len(batches) - 1:
            if batches[i] == batches[i+1]:
                del batches[i+1]
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
                        f'the event size {this_batch_total_length} is greater than {max_token_length}, skip this file, or increase the max token length')
                    this_batch_total_length = 0
                else:
                    batches_new.append([batches[0]])
                    this_batch_total_length = len(batches[0])
            del batches[0]
        del batches
        gc.collect()
        return batches_new

    def random_word(self, events):
        ratios = [self.structure_mask_ratio, self.duration_mask_ratio, self.pitch_mask_ratio,
                  self.control_mask_ratio, self.header_mask_ratio]
        token_types = [structure_token, durations, pitches, control_tokens, header_token]

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
            # bar_pos = np.where(np.array(event) == 'bar')[0]

            # print(len(event))
            # while total_masked_ratio < 0.15 and start_pos < len(event):
            #
            #     masked_token = []
            #     prob = random.random()
            #     if prob < 0.15:
            #
            #     # if prob < 0.083:
            #     #     prob /= 0.083
            #
            #         # # 40% mask 2 event
            #         # if prob < 0.4:
            #         #     if start_pos + 2 <= len(event):
            #         #         masked_token = event[start_pos:start_pos+2]
            #         #         tokens.append(self.vocab.mask_index)
            #         #         total_masked_ratio += 2/len(event)
            #         #         start_pos += 2
            #         #
            #         # # 40% mask 1 token
            #         # elif 0.4 <= prob < 0.8:
            #         #     if start_pos + 1 <= len(event):
            #         #         masked_token = event[start_pos]
            #         #         tokens.append(self.vocab.mask_index)
            #         #         total_masked_ratio += 1/len(event)
            #         #         start_pos += 1
            #         #
            #         # # 20% mask 3 event
            #         # else:
            #         #     if start_pos + 3 <= len(event):
            #         #         masked_token = event[start_pos:start_pos+3]
            #         #         tokens.append(self.vocab.mask_index)
            #         #         total_masked_ratio += 3 / len(event)
            #         #         start_pos += 3
            #
            #         if start_pos + 1 <= len(event):
            #             masked_token = event[start_pos]
            #             tokens.append(self.vocab.mask_index)
            #             total_masked_ratio += 1/len(event)
            #             start_pos += 1
            #
            #         if len(masked_token) > 0:
            #             if not isinstance(masked_token,list):
            #                 masked_token = [masked_token]
            #             decoder_in.append(self.vocab.mask_index)
            #             for token in masked_token:
            #                 decoder_in.append(self.vocab.char2index(token))
            #                 decoder_target.append(self.vocab.char2index(token))
            #             else:
            #                 decoder_target.append(self.vocab.eos_index)
            #
            #         # print(start_pos)
            #         # print(total_masked_ratio)
            #
            #
            #     else:
            #         tokens.append(self.vocab.char2index(event[start_pos]))
            #         start_pos += 1
            #         # print(total_masked_ratio)

            current_track_unmasked = True
            #
            # current_track = 0
            # unmasked = True


            while total_masked_ratio < self.total_mask_ratio and start_pos < len(event):

                prob = random.random()

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


                for i, token_type in enumerate(token_types):
                    if event[start_pos] in token_type:
                        # current_track_unmasked and current_track == 0
                        if prob < ratios[i]:
                            masked_token = event[start_pos]
                            tokens.append(self.vocab.mask_indices[masked_num])
                            total_masked_ratio += 1 / len(event)
                            start_pos += 1



                            if len(masked_token) > 0:
                                if not isinstance(masked_token, list):
                                    masked_token = [masked_token]
                                decoder_in.append(self.vocab.mask_indices[masked_num])
                                for token in masked_token:
                                    decoder_in.append(self.vocab.char2index(token))
                                    decoder_target.append(self.vocab.char2index(token))
                                else:
                                    decoder_target.append(self.vocab.eos_index)


                                # current_track_unmasked = False
                                # masked_num += 1

                                # unmasked  = False
                                break

                        else:
                            tokens.append(self.vocab.char2index(event[start_pos]))
                            start_pos += 1
                            break

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


def collate_mlm(batch):
    input_lens = [x.shape[0] for x in batch[0][0]]
    max_input_len = max(input_lens)

    target_lens = [x.shape[0] for x in batch[0][1]]
    max_target_len = max(target_lens)

    # input
    input_padded = [pad1d(x, max_input_len) for x in batch[0][0]]
    input_padded = np.stack(input_padded)

    input_pad_masks = input_padded == 0

    # target
    target_in_padded = [pad1d(x, max_target_len) for x in batch[0][1]]
    target_in_padded = np.stack(target_in_padded)

    target_in_pad_masks = target_in_padded == 0

    target_out_padded = [pad1d(x, max_target_len) for x in batch[0][2]]
    target_out_padded = np.stack(target_out_padded)

    # print(input_padded.shape)
    # print(target_padded.shape)
    input_pad = torch.tensor(input_padded).long()
    target_in_pad = torch.tensor(target_in_padded).long()
    target_out_pad = torch.tensor(target_out_padded).long()
    input_pad_masks = torch.tensor(input_pad_masks).bool()
    target_in_pad_masks = torch.tensor(target_in_pad_masks).bool()

    output = {"input": input_pad,
              "target_in": target_in_pad,
              "target_out": target_out_pad,
              "input_pad_mask": input_pad_masks,
              "target_pad_mask": target_in_pad_masks
              }

    return output

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

# #
# vocab = WordVocab(all_tokens)
# event_folder = '../dataset/lmd/lmd_separated_event/'
# file_size = 100
# window_size=8
#
# dataset = ParallelLanguageDataset(event_folder, vocab, 0,1,2000, window_size)
# data_loader = DataLoader(dataset, batch_size=8, collate_fn=lambda batch: collate_mlm(batch))  # 训练语料按长度排好序的
#
# for i in data_loader:
#     print(i)

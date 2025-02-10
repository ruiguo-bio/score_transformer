import pickle
import numpy as np


pad = '<pad>'
eos = '<eos>'
mask = [f'm_{num}' for num in range(1)]

special_tokens = [pad, eos]

time_signature_token = ['4/4', '3/4', '2/4', '6/8']

program_num = [f'i_{num}' for num in range(128)]

tempo_token = [f't_{i}' for i in range(7)]

track_num = [f'track_{num}' for num in range(3)]

step_token = [f'e_{num}' for num in range(16)]

duration_single = [f'n_{num}' for num in range(1,33)]
duration_tokens = step_token + duration_single
structure_token = ['bar'] + track_num

song_token = time_signature_token + tempo_token
header_token = time_signature_token + tempo_token + program_num
header_without_key_token = time_signature_token + tempo_token + program_num


pitch_tokens = [f'p_{num}' for num in range(21, 109)]
pitches = pitch_tokens

note_tokens = pitches + duration_tokens

all_tokens = special_tokens + mask + structure_token + \
             header_token  + note_tokens

control_bins = np.arange(0, 1, 0.1)
tensile_bins = np.arange(0, 2.1, 0.2).tolist() + [4]
diameter_bins = np.arange(0, 4.1, 0.4).tolist() + [5]
tempo_bins = np.array([0] + list(range(60, 190, 30)) + [210])

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
        self.name_to_tokens= {}
        self.structure_indices = [self._char2idx[name] for name in structure_token]
        self.pitch_indices = [self._char2idx[name] for name in pitches]
        self.mask_indices = [self._char2idx[name] for name in mask]

        self.duration_indices = [self._char2idx[name] for name in duration_tokens]

        self.step_indices = [self._char2idx[name] for name in step_token]
        self.duration_only_indices = [self._char2idx[name] for name in duration_single]

        self.program_indices = [self._char2idx[name] for name in program_num]
        self.tempo_indices = [self._char2idx[name] for name in tempo_token]
        self.time_signature_indices = [self._char2idx[name] for name in time_signature_token]


        for index in self.program_indices:
            self.token_class_ranges[index] = 'program'
            if 'program' in self.name_to_tokens:
                self.name_to_tokens['program'].append(self._idx2char[index])
            else:
                self.name_to_tokens['program'] = [self._idx2char[index]]
        for index in self.tempo_indices:
            self.token_class_ranges[index] = 'tempo'
            if 'tempo' in self.name_to_tokens:
                self.name_to_tokens['tempo'].append(self._idx2char[index])
            else:
                self.name_to_tokens['tempo'] = [self._idx2char[index]]
        for index in self.time_signature_indices:
            self.token_class_ranges[index] = 'time_signature'
            if 'time_signature' in self.name_to_tokens:
                self.name_to_tokens['time_signature'].append(self._idx2char[index])
            else:
                self.name_to_tokens['time_signature'] = [self._idx2char[index]]

        for index in self.structure_indices:
            self.token_class_ranges[index] = 'structure'
            if 'structure' in self.name_to_tokens:
                self.name_to_tokens['structure'].append(self._idx2char[index])
            else:
                self.name_to_tokens['structure'] = [self._idx2char[index]]
        for index in self.pitch_indices:
            self.token_class_ranges[index] = 'pitch'
            if 'pitch' in self.name_to_tokens:
                self.name_to_tokens['pitch'].append(self._idx2char[index])
            else:
                self.name_to_tokens['pitch'] = [self._idx2char[index]]
        for index in self.duration_indices:
            self.token_class_ranges[index] = 'duration'
            if 'duration' in self.name_to_tokens:
                self.name_to_tokens['duration'].append(self._idx2char[index])
            else:
                self.name_to_tokens['duration'] = [self._idx2char[index]]

        self.token_class_ranges[self.eos_index] = 'eos'

        self.name_to_tokens['eos'] = self._idx2char[self.eos_index]
        self.class_names = set(self.token_class_ranges.values())


    def char2index(self, token):
        if token not in self._char2idx:
            print('invalid')

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

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

structure_token = ['bar'] + track_num

all_key_names = ['C major', 'G major', 'D major', 'A major',
                 'E major', 'B major', 'F major', 'B- major',
                 'E- major', 'A- major', 'D- major', 'G- major',
                 'A minor', 'E minor', 'B minor', 'F# minor',
                 'C# minor', 'G# minor', 'D minor', 'G minor',
                 'C minor', 'F minor', 'B- minor', 'E- minor',
                 ]

key_token = [f'k_{num}' for num in range(len(all_key_names))]
key_to_token = {name: f'k_{i}' for i, name in enumerate(all_key_names)}
token_to_key = {v: k for k, v in key_to_token.items()}

header_token = time_signature_token + tempo_token + program_num + key_token

track_note_density_token = [f'd_{num}' for num in range(10)]
track_occupation_rate_token = [f'o_{num}' for num in range(10)]
track_polyphony_rate_token = [f'y_{num}' for num in range(10)]
track_pitch_register_token = [f'r_{num}' for num in range(10)]

track_control_tokens = track_note_density_token + \
                       track_occupation_rate_token + \
                       track_polyphony_rate_token + \
                       track_pitch_register_token

tensile_strain_token = [f's_{num}' for num in range(12)]
diameter_token = [f'a_{num}' for num in range(12)]

bar_control_tokens = tensile_strain_token + diameter_token

control_tokens = bar_control_tokens + track_control_tokens + key_token

rests = ['rest_e', 'rest_s']

durations = ['whole', 'half', 'quarter', 'eighth', 'sixteenth']

pitches = [f'p_{num}' for num in range(21, 109)] + rests + ['continue']

note_tokens = pitches + durations

all_tokens = special_tokens + mask + structure_token + \
             header_token + track_control_tokens + bar_control_tokens + \
             note_tokens

control_bins = np.arange(0, 1, 0.1)
# tensile_bins = np.arange(0, 2, 0.1).tolist() + np.arange(2, 2.8, 0.2).tolist() + [4]
tensile_bins = np.arange(0, 2.1, 0.2).tolist() + [4]
# diameter_bins = np.arange(0, 4.8, 0.2).tolist() + [5]
diameter_bins = np.arange(0, 4.1, 0.4).tolist() + [5]
tempo_bins = np.array([0] + list(range(60, 190, 30)) + [1000])


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
        self.key_indices = [self._char2idx[name] for name in key_token]
        self.density_indices = [self._char2idx[name] for name in track_note_density_token]
        self.occupation_indices = [self._char2idx[name] for name in track_occupation_rate_token]
        self.polyphony_indices = [self._char2idx[name] for name in track_polyphony_rate_token]
        self.pitch_register_indices = [self._char2idx[name] for name in track_pitch_register_token]
        self.program_indices = [self._char2idx[name] for name in program_num]
        self.tempo_indices = [self._char2idx[name] for name in tempo_token]
        self.time_signature_indices = [self._char2idx[name] for name in time_signature_token]
        self.tensile_indices = [self._char2idx[name] for name in tensile_strain_token]
        self.diameter_indices = [self._char2idx[name] for name in diameter_token]
        self.control_indices = self.key_indices + self.density_indices + self.occupation_indices + \
                               self.polyphony_indices + self.pitch_register_indices + self.tempo_indices + \
                               self.tensile_indices + self.diameter_indices + self.program_indices

        self.control_names = ['program', 'tempo', 'key', 'pitch_register',
                              'polyphony', 'density', 'occupation', 'tensile',
                              'diameter']
        for index in self.program_indices:
            self.token_class_ranges[index] = 'program'
        for index in self.tempo_indices:
            self.token_class_ranges[index] = 'tempo'
        for index in self.time_signature_indices:
            self.token_class_ranges[index] = 'time_signature'
        for index in self.key_indices:
            self.token_class_ranges[index] = 'key'
        for index in self.density_indices:
            self.token_class_ranges[index] = 'density'
        for index in self.occupation_indices:
            self.token_class_ranges[index] = 'occupation'
        for index in self.polyphony_indices:
            self.token_class_ranges[index] = 'polyphony'
        for index in self.pitch_register_indices:
            self.token_class_ranges[index] = 'pitch_register'

        for index in self.structure_indices:
            self.token_class_ranges[index] = 'structure'
        for index in self.pitch_indices:
            self.token_class_ranges[index] = 'pitch'
        for index in self.duration_indices:
            self.token_class_ranges[index] = 'duration'
        # for index in self.mask_indices:
        #     self.token_class_ranges[index] = 'mask'

        for index in self.tensile_indices:
            self.token_class_ranges[index] = 'tensile'
        for index in self.diameter_indices:
            self.token_class_ranges[index] = 'diameter'

        self.token_class_ranges[self.eos_index] = 'eos'
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

from pathlib import Path
import numpy as np
import kaldiio
from tqdm import tqdm
import torch

from utils import read_kaldi_format, save_kaldi_format, setup_logger

logger = setup_logger(__name__)

class SpeechEmbeddings:

    def __init__(self):
        self.embeddings = []
        # self.vectors = []       #actual embeddings
        self.utterances = []
        self.speakers = []
        self.utt2idx = {}

        self.new = True
        # self.is_phones = is_phones

    def __len__(self):
        return len(self.utterances)

    def __iter__(self):
        for i in range(len(self)):
            yield self.embeddings[i], self.utterances[i], self.speakers[i]

    def __getitem__(self, utt):
        return self.get_instance(utt)[0]

    def __contains__(self, item):
        return item in self.utterances

    # def write_to_ark(self, vectors_data, out_dir: Path, tqdm_desc="Writing to ARK"):
    #     # Use kaldiio.save_ark to write both .ark and .scp
    #     ark_filename = str(out_dir) + '.ark'
    #     scp_filename = str(out_dir) + '.scp'

    #     print(f"Now saving embeddings to {ark_filename}")
    #     if not vectors_data:
    #         logger.warning("No data provided to write_to_ark. Skipping.")
    #         return

    #     # Prepare data for kaldiio.save_ark and for populating self.
    #     kaldi_data_dict = {}
    #     new_utterances = []
    #     new_speakers = []
    #     new_embeddings_for_self = [] # This will store the actual numpy arrays

    #     # Sort for consistent .scp output, crucial for Kaldi tools
    #     # sorted_vectors_data = sorted(vectors_data, key=lambda x: x['utt'])

    #     for item in tqdm(vectors_data, desc=tqdm_desc):
    #         utt_id = item['utt']
    #         spk_id = item['spk']
    #         token_vector = item['token'] # This should be a numpy array of integers

    #         if not isinstance(token_vector, np.ndarray) or not np.issubdtype(token_vector.dtype, np.integer):
    #             print(type(token_vector))
    #             logger.error(f"Token vector for {utt_id} is not a numpy array of integers. Type: {type(token_vector)}, Dtype: {token_vector.dtype}. Please ensure it's converted.")
    #             raise TypeError("Token vector must be a numpy array of integers.")

    #         kaldi_data_dict[utt_id] = token_vector
    #         new_utterances.append(utt_id)
    #         new_speakers.append(spk_id)
    #         new_embeddings_for_self.append(token_vector) # Store the actual vector in memory

    #     try:
    #         kaldiio.save_ark(ark_filename, kaldi_data_dict, scp=scp_filename, write_function="numpy")
    #         logger.info(f"Successfully saved discrete units to {ark_filename} and {scp_filename}")
    #         self.new = False # Indicate that data has been saved/processed

    #         # Update internal state (embeddings will now store the keys/utt_ids)
    #         # This is critical: self.embeddings should store the *keys* that map to the ARK file,
    #         # not the actual vectors themselves if you intend to only load them on demand.
    #         # However, your add_instance takes 'embedding' (which I assume is the actual vector).
    #         # Let's adjust self.embeddings to store the utt_id keys, and self.vectors to store actual data.
    #         # This makes __getitem__ work directly if data is loaded into self.vectors.

    #         # Clear existing data if this is a new write operation
    #         self.embeddings = [] # This will store utt_ids (keys to the ark)
    #         self.vectors = []    # This will store the actual np arrays in memory
    #         self.utterances = []
    #         self.speakers = []
    #         self.utt2idx = {}

    #         # Add the newly processed data
    #         self.add_instances(new_embeddings_for_self, new_utterances, new_speakers)


    #     except Exception as e:
    #         logger.error(f"An error occurred during Kaldi IO with kaldiio.save_ark: {e}")
    #         raise # Re-raise the exception

    def add_instance(self, embedding, utterance, speaker):
        self.utt2idx[utterance] = len(self)
        self.embeddings.append(embedding)
        self.utterances.append(utterance)
        self.speakers.append(speaker)

    def add_instances(self, embeddings, utterances, speakers):
        if len(set(utterances) & set(self.utterances)) > 0:
            remove_indices = [i for i, utt in enumerate(utterances) if utt in self.utterances]
            for idx in reversed(remove_indices):
                del embeddings[idx]
                del utterances[idx]
                del speakers[idx]

        self.utt2idx.update({utt: len(self) + i for i, utt in enumerate(utterances)})
        self.embeddings.extend(embeddings)
        self.utterances.extend(utterances)
        self.speakers.extend(speakers)

    def get_instance(self, utterance):
        idx = self.utt2idx[utterance]
        return self.embeddings[idx], self.speakers[idx]

    # def remove_special_characters(self, embedding):
    #     return embedding.replace('«', '').replace('»', '').replace('^', '')

    def get_iterators(self, n):
        # divides the stored data into n packages and returns a list of iterators over each package
        # like __iter__, but with several iterators

        def _get_instance_by_indices(indices):
            for i in indices:
                yield self.embeddings[i], self.utterances[i], self.speakers[i]

        iterator_length = len(self) // n
        it_slices = [[iterator_length * i, iterator_length * (i + 1)] if i < (n - 1)
                     else [iterator_length * i, len(self)] for i in range(n)]
        iterators = [_get_instance_by_indices(range(*it_slice)) for it_slice in it_slices]
        return iterators

    # def update_instance(self, utterance, embedding):
    #     idx = self.utt2idx[utterance]
    #     self.embeddings[idx] = self.remove_special_characters(embedding)

    def remove_instances(self, utterance_list):
        remove_indices = [self.utt2idx[utt] for utt in utterance_list if utt in self.utt2idx]
        remove_indices = sorted(remove_indices, reverse=True)
        for idx in remove_indices:
            del self.embeddings[idx]
            del self.utterances[idx]
            del self.speakers[idx]
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}

    def get_instances_of_speaker(self, speaker):
        indices = [i for (i, spk) in enumerate(self.speakers) if spk == speaker]
        embeddings = [self.embeddings[i] for i in indices]
        utterances = [self.utterances[i] for i in indices]
        return embeddings, utterances

    def shuffle(self):
        shuffled_embeddings = []
        shuffled_utterances = []
        shuffled_speakers = []
        for i in np.random.permutation(len(self)):
            shuffled_embeddings.append(self.embeddings[i])
            shuffled_utterances.append(self.utterances[i])
            shuffled_speakers.append(self.speakers[i])
        self.embeddings = shuffled_embeddings
        self.utterances = shuffled_utterances
        self.speakers = shuffled_speakers
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}

    def save_tokens(self, out_dir: Path, add_suffix=None):
        out_dir.mkdir(exist_ok=True, parents=True)
        add_suffix = add_suffix if add_suffix is not None else ""
        # save_kaldi_format(data=list(zip(self.utterances, self.embeddings)),
        #                   filename=out_dir / f'token{add_suffix}')
        torch.save(list(zip(self.utterances, self.embeddings)), out_dir / f'token{add_suffix}')
        save_kaldi_format(data=list(zip(self.utterances, self.speakers)),
                          filename=out_dir / f'utt2spk{add_suffix}')

    def load_tokens(self, in_dir, add_suffix=None):
        self.new = False
        add_suffix = add_suffix if add_suffix is not None else ""
        # utt_1, embeddings = read_kaldi_format(filename=in_dir / f'token{add_suffix}', return_as_dict=False,
        #                                      values_as_string=True)
        # embeddings = [self.remove_special_characters(embedding) for embedding in embeddings]

        token_data = torch.load(in_dir / f'token{add_suffix}')

        utt_1 = [item[0] for item in token_data]
        embeddings = [item[1] for item in token_data]

        utt_2, speakers = read_kaldi_format(filename=in_dir / f'utt2spk{add_suffix}', return_as_dict=False)

        if utt_1 == utt_2:
            self.utterances = utt_1
            self.embeddings = embeddings
            self.speakers = speakers
        elif sorted(utt_1) == sorted(utt_2):
            self.utterances, self.embeddings = zip(*sorted(zip(utt_1, embeddings), key=lambda x: x[0]))
            _, self.speakers = zip(*sorted(zip(utt_2, speakers), key=lambda x: x[0]))
        else:
            raise ValueError(f'{in_dir / f"emb{add_suffix}"} and {in_dir / f"utt2spk{add_suffix}"} have mismatching '
                             f'utterance keys; embeddings cannot be loaded!')
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}

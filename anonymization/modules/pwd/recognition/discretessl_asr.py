from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoProcessor, HubertModel, pipeline
from speechbrain.lobes.models.huggingface_transformers.hubert import (HuBERT)
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL
from datasets import load_dataset
from ..speech_embeddings import SpeechEmbeddings


class DiscreteSSLASR:

    def __init__(self, model_path, device, utt_start_token='', utt_end_token='', lang=None, batch_size=16, **kwargs):
        self.device = device
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_path = model_path  # TODO
        self.use_flash_attention_2 = False
        self.utt_start_token = utt_start_token
        self.utt_end_token = utt_end_token
        self.lang = lang

        # model_id = "facebook/hubert-large-ls960-ft"
        # # model_id = "utter-project/mHuBERT-147"
        # model = HubertModel.from_pretrained(model_id, cache_dir=model_path, torch_dtype=torch_dtype)
        # model.to(self.device)
        # processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_path)
        # # processor = Wav2Vec2Processor.from_pretrained(model_id, cache_dir=model_path)
        # self.speech2emb = pipeline('feature-extraction', model=model, feature_extractor=processor.feature_extractor,
        #                            tokenizer=processor.tokenizer, batch_size=batch_size,
        #                             torch_dtype=torch_dtype, device=self.device,  max_new_tokens=128, framework='pt')

        # self.outputs = []
        # self.tokens = []

        # Below is DiscreteSSl use
        model_id = "facebook/hubert-large-ll60k"
        # model_id = "utter-project/mHuBERT-147"
        self.ssl_layer_num = [7,23]
        self.deduplicate =[False, True]
        self.bpe_tokenizers=[None, None]
        vocoder_repo_id = "speechbrain/hifigan-hubert-k1000-LibriTTS"
        kmeans_dataset = "LibriSpeech"
        num_clusters = 1000
        ssl_model = HuBERT(model_id, model_path, output_all_hiddens=True)

        self.speech2emb = DiscreteSSL(model_path, ssl_model, vocoder_repo_id=vocoder_repo_id, kmeans_dataset=kmeans_dataset, num_clusters=num_clusters)
        self.tokens = []
        # self.embeddings = 'embeddings'
        # self.processed_tokens = 'processed_tokens'
        self.outputs = 'outputs'

    # def _extract_continuous_features(self, audio_path, target_sr=16000):
    #     """Helper to load audio and extract last hidden states from HuBERT."""
    #     # try:
    #     #     waveform, sr = torchaudio.load(audio_path)
    #     # except Exception as e:
    #     #     logger.warning(f"Could not load audio from {audio_path}: {e}")
    #     #     return None

    #     # if sr != target_sr:
    #     #     waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

    #     # if waveform.shape[0] > 1: # Ensure mono
    #     #     waveform = waveform.mean(dim=0, keepdim=True)

    #     # input_values = self.processor(waveform.squeeze(0).numpy(), sampling_rate=target_sr, return_tensors="pt").input_values
    #     # input_values = input_values.to(self.device)

    #     with torch.no_grad():
    #         outputs = self.model(input_values, output_hidden_states=True)
    #         # hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    #         hidden_states = outputs.hidden_states

    #     return hidden_states

    def recognize_speech_of_audio(self, audio_file):
        # tokens, _,_ = self.speech2emb.encode(audio_file,
                                    # SSL_layers=self.ssl_layer_num, deduplicates=self.deduplicate, bpe_tokenizers=self.bpe_tokenizers)
        tokens = self.speech2emb(audio_file)
        # text = self.speech2text(audio_file)['text']
        # text = self.utt_start_token + text.strip() + self.utt_end_token
        return tokens


    def recognize_speech_of_dataset(self, audio_dataset, out_dir, save_intermediate=True, job_id=None):
        # texts = Text(is_phones=(self.output == 'phones'))

        embs = SpeechEmbeddings()

        # tokens = []
        # outputs,_,_ = self.speech2emb.encode(audio_dataset,
        #                             SSL_layers=self.ssl_layer_num, deduplicates=self.deduplicate, bpe_tokenizers=self.bpe_tokenizers)

        utts, spks, wav_files, wav_lens = audio_dataset.audios()
        print(f'length of audio_dataset={len(utts)}')
        tokens, _, _ = self.speech2emb.encode(wav_files, wav_lens,
                            SSL_layers=self.ssl_layer_num, deduplicates=self.deduplicate, bpe_tokenizers=self.bpe_tokenizers)

        outputs = [{'utt':utt, 'spk':spk, 'token':token} for utt,spk,token in zip(utts, spks, tokens)]

        if job_id is None:  # single processing
            add_suffix = None
            tqdm_params = {}
        else: # process amongst multiple processes
            add_suffix = f'_{job_id}'
            tqdm_params = {'desc': f'Job {job_id}', 'leave': True}

        i = 0
        for output in tqdm(outputs, **tqdm_params):
            utt = output['utt']
            spk = output['spk']
            token = output['token']
            embs.add_instance(output, utterance=utt, speaker=spk)

            i += 1
            if i % 100 == 0 and save_intermediate:
                embs.save_tokens(out_dir=out_dir, add_suffix=add_suffix)

        if save_intermediate:
            embs.save_tokens(out_dir=out_dir, add_suffix=add_suffix)

        return embs

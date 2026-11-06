import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT
from speechbrain.lobes.models.huggingface_transformers.discrete_ssl import DiscreteSSL
import torchaudio
from sklearn.cluster import MiniBatchKMeans
import os
import joblib
from glob import glob
from tqdm import tqdm
import numpy as np

# Step 1: Extract hidden states from specific layers

# Settings
model_id = "facebook/hubert-large-ll60k"
device = "cuda" if torch.cuda.is_available() else "cpu"
target_layers = [18, 19, 20, 21, 22, 23]
num_clusters = 1000
model_path = "./kmeans_top_layers_model"

# Here's the path to all datasets, but how to load it stagewise (?)
dataset_folder = "/vil/spk-anon-foundation-models/data"

# Load model + processor
model = HubertModel.from_pretrained(model_id, output_hidden_states=True, cache_dir=model_path).to(device)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, cache_dir=model_path)

os.makedirs(model_path, exist_ok=True)

def load_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform.squeeze(0)


def extract_features(file_list, layers):
    layer_features = {layer: [] for layer in layers}

    for file_path in tqdm(file_list, desc="Extracting features"):
        audio = load_audio(file_path).numpy()
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(inputs.input_values.to(device))
            hidden_states = outputs.hidden_states  # List of (batch, seq, hidden_dim)

        for layer in layers:
            h = hidden_states[layer].squeeze(0).cpu().numpy()  # (seq_len, dim)
            layer_features[layer].append(h)

    return layer_features


# Load dataset
file_list = glob(os.path.join(dataset_folder, "**/*.wav"), recursive=True)  # Adjust logic to navigate dirs
features_by_layer = extract_features(file_list, target_layers)

# Step 2: Fit K-Means model per layer

def train_kmeans(layer_features, layer, k=1000):
    data = np.concatenate(layer_features[layer], axis=0)  # Shape: (N, dim)
    print(f"Training k-means for layer {layer} with shape: {data.shape}")

    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, verbose=1)
    kmeans.fit(data)

    model_filename = os.path.join(model_path, f"kmeans/LibriSpeech_hubert_k{k}_L{layer}.pt")
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(kmeans, model_filename)
    print(f"Saved k-means for layer {layer} to {model_filename}")


# Train and save
for layer in target_layers:
    train_kmeans(features_by_layer, layer, k=num_clusters)

# Step 3: Make new layers available to DiscreteSSL
# No need to modify the original class DiscreteSSL if models are saved in the same naming pattern
# as described in the DiscreteSSL class in discrete_ssl.py, for example:
# kmeans/LibriSpeech_hubert_k1000_L18.pt
# kmeans/LibriSpeech_hubert_k1000_L23.pt

# Telling DiscreteSSL to load the models
ssl_layer_num = [18, 19, 20, 21, 22, 23]
deduplicates = [False] * len(ssl_layer_num)
bpe_tokenizers = [None] * len(ssl_layer_num)

ssl_model = HuBERT(
    model_hub=model_id,
    save_path=model_path,
    output_all_hiddens=True,
)

speech2emb = DiscreteSSL(
    save_path=model_path,
    ssl_model=ssl_model,
    vocoder_repo_id="speechbrain/hifigan-hubert-k1000-LibriTTS",
    kmeans_dataset="LibriSpeech",
    num_clusters=[num_clusters] * len(ssl_layer_num),
    layers_num=ssl_layer_num,
    device=device,
)

print("DiscreteSSL initialized with custom k-means models.")

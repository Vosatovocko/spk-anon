import glob
import os
import random
from pathlib import Path


def limit_to_n(path_to_transcript_dict, n=40000):
    limited_dict = dict()
    if len(path_to_transcript_dict.keys()) > n:
        for key in random.sample(list(path_to_transcript_dict.keys()), n):
            limited_dict[key] = path_to_transcript_dict[key]
        return limited_dict
    else:
        return path_to_transcript_dict


def build_path_to_transcript_dict_mls_italian():
    lang = "italian"
    root = f"data/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(build_path_to_transcript_dict_multi_ling_librispeech_template(root=root))


def build_path_to_transcript_dict_mls_french():
    lang = "french"
    root = f"data/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(build_path_to_transcript_dict_multi_ling_librispeech_template(root=root))


def build_path_to_transcript_dict_mls_dutch():
    lang = "dutch"
    root = f"data/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(build_path_to_transcript_dict_multi_ling_librispeech_template(root=root))


def build_path_to_transcript_dict_mls_polish():
    lang = "polish"
    root = f"data/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(build_path_to_transcript_dict_multi_ling_librispeech_template(root=root))


def build_path_to_transcript_dict_mls_spanish():
    lang = "spanish"
    root = f"data/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(build_path_to_transcript_dict_multi_ling_librispeech_template(root=root))


def build_path_to_transcript_dict_mls_portuguese():
    lang = "portuguese"
    root = f"data/corpora/MultiLingLibriSpeech/mls_{lang}/train"
    return limit_to_n(build_path_to_transcript_dict_multi_ling_librispeech_template(root=root))


def build_path_to_transcript_dict_multi_ling_librispeech_template(root):
    """
    https://arxiv.org/abs/2012.03411
    """
    path_to_transcript = dict()
    with open(os.path.join(root, "transcripts.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_folders = line.split("\t")[0].split("_")
            wav_path = os.path.join(root, "audio", wav_folders[0], wav_folders[1], line.split("\t")[0] + ".flac")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
            else:
                print(f"not found: {wav_path}")
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_karlsson():
    root = "data/corpora/HUI_German/Karlsson"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_eva():
    root = "data/corpora/HUI_German/Eva"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_bernd():
    root = "data/corpora/HUI_German/Bernd"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_friedrich():
    root = "data/corpora/HUI_German/Friedrich"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hokus():
    root = "data/corpora/HUI_German/Hokus"
    return limit_to_n(build_path_to_transcript_dict_hui_template(root=root))


def build_path_to_transcript_dict_hui_others():
    root = "data/corpora/HUI_German/others"
    pttd = dict()
    for speaker in os.listdir(root):
        pttd.update(build_path_to_transcript_dict_hui_template(root=f"{root}/{speaker}"))
    return pttd


def build_path_to_transcript_dict_hui_template(root):
    """
    https://arxiv.org/abs/2106.06309
    """
    path_to_transcript = dict()
    for el in os.listdir(root):
        if os.path.isdir(os.path.join(root, el)):
            with open(os.path.join(root, el, "metadata.csv"), "r", encoding="utf8") as file:
                lookup = file.read()
            for line in lookup.split("\n"):
                if line.strip() != "":
                    norm_transcript = line.split("|")[1]
                    wav_path = os.path.join(root, el, "wavs", line.split("|")[0] + ".wav")
                    if os.path.exists(wav_path):
                        path_to_transcript[wav_path] = norm_transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_elizabeth():
    root = "data/corpora/MAILabs_british_single_speaker_elizabeth"
    path_to_transcript = dict()
    for el in os.listdir(root):
        if os.path.isdir(os.path.join(root, el)):
            with open(os.path.join(root, el, "metadata.csv"), "r", encoding="utf8") as file:
                lookup = file.read()
            for line in lookup.split("\n"):
                if line.strip() != "":
                    norm_transcript = line.split("|")[2]
                    wav_path = os.path.join(root, el, "wavs", line.split("|")[0] + ".wav")
                    if os.path.exists(wav_path):
                        path_to_transcript[wav_path] = norm_transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_nancy():
    root = "data/corpora/NancyKrebs"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("|")[1]
            wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_integration_test():
    root = "data/corpora/NancyKrebs"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n")[:500]:
        if line.strip() != "":
            norm_transcript = line.split("|")[1]
            wav_path = os.path.join(root, "wav", line.split("|")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_hokuspokus():
    path_to_transcript = dict()
    for transcript_file in os.listdir("data/corpora/LibriVox.Hokuspokus/txt"):
        if transcript_file.endswith(".txt"):
            with open("data/corpora/LibriVox.Hokuspokus/txt/" + transcript_file, 'r',
                      encoding='utf8') as tf:
                transcript = tf.read()
            wav_path = "data/corpora/LibriVox.Hokuspokus/wav/" + transcript_file.rstrip(
                ".txt") + ".wav"
            path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_fluxsing():
    root = "data/corpora/FluxSing"
    path_to_transcript = dict()
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("|")[2]
            wav_path = os.path.join(root, line.split("|")[0])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_vctk():
    path_to_transcript = dict()
    for transcript_dir in os.listdir("data/corpora/VCTK/txt"):
        for transcript_file in os.listdir(f"data/corpora/VCTK/txt/{transcript_dir}"):
            if transcript_file.endswith(".txt"):
                with open(f"data/corpora/VCTK/txt/{transcript_dir}/" + transcript_file, 'r',
                          encoding='utf8') as tf:
                    transcript = tf.read()
                wav_path = f"data/corpora/VCTK/wav48_silence_trimmed/{transcript_dir}/" + transcript_file.rstrip(
                    ".txt") + "_mic2.flac"
                if os.path.exists(wav_path):
                    path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_libritts():
    path_train = "data/corpora/LibriTTS/train-clean-100"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_libritts_all_clean():
    path_train = "data/corpora/LibriTTS/all_clean"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
    return path_to_transcript


def build_path_to_transcript_dict_libritts_other500():
    path_train = "data/asr-data/LibriTTS/train-other-500"
    path_to_transcript = dict()
    for speaker in os.listdir(path_train):
        for chapter in os.listdir(os.path.join(path_train, speaker)):
            for file in os.listdir(os.path.join(path_train, speaker, chapter)):
                if file.endswith("normalized.txt"):
                    with open(os.path.join(path_train, speaker, chapter, file), 'r', encoding='utf8') as tf:
                        transcript = tf.read()
                    wav_file = file.split(".")[0] + ".wav"
                    path_to_transcript[os.path.join(path_train, speaker, chapter, wav_file)] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_ljspeech():
    path_to_transcript = dict()
    for transcript_file in os.listdir("data/corpora/LJSpeech/16kHz/txt"):
        with open("data/corpora/LJSpeech/16kHz/txt/" + transcript_file, 'r', encoding='utf8') as tf:
            transcript = tf.read()
        wav_path = "data/corpora/LJSpeech/16kHz/wav/" + transcript_file.rstrip(".txt") + ".wav"
        path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_att_hack():
    path_to_transcript = dict()
    for transcript_file in os.listdir("data/corpora/FrenchExpressive/txt"):
        if transcript_file.endswith(".txt"):
            with open("data/corpora/FrenchExpressive/txt/" + transcript_file, 'r',
                      encoding='utf8') as tf:
                transcript = tf.read()
            wav_path = "data/corpora/FrenchExpressive/wav/" + transcript_file.rstrip(".txt") + ".wav"
            path_to_transcript[wav_path] = transcript
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10de():
    path_to_transcript = dict()
    with open("data/corpora/CSS10/german/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript["data/corpora/CSS10/german/" + line.split("|")[0]] = line.split("|")[
                2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10cmn():
    path_to_transcript = dict()
    with open("data/corpora/CSS10/chinese/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript["data/corpora/CSS10/chinese/" + line.split("|")[0]] = line.split("|")[
                2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_vietTTS():
    path_to_transcript = dict()
    root = "data/corpora/VietTTS"
    with open(root + "/meta_data.tsv", encoding="utf8") as f:
        transcriptions = f.read()
    for line in transcriptions.split("\n"):
        if line.strip() != "":
            parsed_line = line.split(".wav")
            audio_path = parsed_line[0]
            transcript = parsed_line[1]
            path_to_transcript[os.path.join(root, audio_path + ".wav")] = transcript.strip()
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_thorsten():
    path_to_transcript = dict()
    root = "data/corpora/Thorsten_DE/V2"
    with open(root + "/metadata_train.csv", encoding="utf8") as f:
        transcriptions = f.read()
    with open(root + "/metadata_dev.csv", encoding="utf8") as f:
        transcriptions += "\n" + f.read()
    with open(root + "/metadata_test.csv", encoding="utf8") as f:
        transcriptions += "\n" + f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[root + "/wavs/" + line.split("|")[0] + ".wav"] = \
                line.split("|")[1]
    return path_to_transcript


def build_path_to_transcript_dict_thorsten_2020():
    path_to_transcript = dict()
    with open("data/corpora/Thorsten_DE/metadata_shuf.csv", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript["data/corpora/Thorsten_DE/wavs/" + line.split("|")[0] + ".wav"] = \
                line.split("|")[1]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10el():
    path_to_transcript = dict()
    language = "greek"
    with open(f"data/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"data/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10nl():
    path_to_transcript = dict()
    language = "dutch"
    with open(f"data/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"data/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10fi():
    path_to_transcript = dict()
    language = "finnish"
    with open(f"data/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"data/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10ru():
    path_to_transcript = dict()
    language = "russian"
    with open(f"data/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"data/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10hu():
    path_to_transcript = dict()
    language = "hungarian"
    with open(f"data/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"data/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10es():
    path_to_transcript = dict()
    language = "spanish"
    with open(f"data/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"data/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_css10fr():
    path_to_transcript = dict()
    language = "french"
    with open(f"data/corpora/CSS10/{language}/transcript.txt", encoding="utf8") as f:
        transcriptions = f.read()
    trans_lines = transcriptions.split("\n")
    for line in trans_lines:
        if line.strip() != "":
            path_to_transcript[f"data/corpora/CSS10/{language}/{line.split('|')[0]}"] = \
                line.split("|")[2]
    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_nvidia_hifitts():
    path_to_transcript = dict()
    transcripts = list()
    root = "data/corpora/hi_fi_tts_v0"

    import json

    for jpath in [f"{root}/6097_manifest_clean_dev.json",
                  f"{root}/6097_manifest_clean_test.json",
                  f"{root}/6097_manifest_clean_train.json",
                  f"{root}/9017_manifest_clean_dev.json",
                  f"{root}/9017_manifest_clean_test.json",
                  f"{root}/9017_manifest_clean_train.json",
                  f"{root}/92_manifest_clean_dev.json",
                  f"{root}/92_manifest_clean_test.json",
                  f"{root}/92_manifest_clean_train.json"]:
        with open(jpath, encoding='utf-8', mode='r') as jfile:
            for line in jfile.read().split("\n"):
                if line.strip() != "":
                    transcripts.append(json.loads(line))

    for transcript in transcripts:
        path = transcript["audio_filepath"]
        norm_text = transcript["text_normalized"]
        path_to_transcript[f"{root}/{path}"] = norm_text

    return limit_to_n(path_to_transcript)


def build_path_to_transcript_dict_spanish_blizzard_train():
    root = "data/corpora/Blizzard2021/spanish_blizzard_release_2021_v2/hub"
    path_to_transcript = dict()
    with open(os.path.join(root, "train_text.txt"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, "train_wav", line.split("\t")[0] + ".wav")
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_aishell3():
    root = "data/corpora/aishell3/train"
    path_to_transcript_dict = dict()
    with open(root + "/label_train-set.txt", mode="r", encoding="utf8") as f:
        transcripts = f.read().replace("$", "").replace("%", ",").split("\n")
    for transcript in transcripts:
        if transcript.strip() != "" and not transcript.startswith("#"):
            parsed_line = transcript.split("|")
            audio_file = f"{root}/wav/{parsed_line[0][:7]}/{parsed_line[0]}.wav"
            kanji = parsed_line[2]
            path_to_transcript_dict[audio_file] = kanji
    return limit_to_n(path_to_transcript_dict)


def build_path_to_transcript_dict_VIVOS_viet():
    root = "data/corpora/VIVOS_vietnamese/train"
    path_to_transcript_dict = dict()
    with open(root + "/prompts.txt", mode="r", encoding="utf8") as f:
        transcripts = f.read().split("\n")
    for transcript in transcripts:
        if transcript.strip() != "":
            parsed_line = transcript.split(" ")
            audio_file = f"{root}/waves/{parsed_line[0][:10]}/{parsed_line[0]}.wav"
            path_to_transcript_dict[audio_file] = " ".join(parsed_line[1:]).lower()
    return limit_to_n(path_to_transcript_dict)


def build_path_to_transcript_dict_RAVDESS():
    root = "data/corpora/RAVDESS"
    path_to_transcript_dict = dict()
    for speaker_dir in os.listdir(root):
        for audio_file in os.listdir(os.path.join(root, speaker_dir)):
            if audio_file.split("-")[4] == "01":
                path_to_transcript_dict[os.path.join(root, speaker_dir, audio_file)] = "Kids are talking by the door."
            else:
                path_to_transcript_dict[os.path.join(root, speaker_dir, audio_file)] = "Dogs are sitting by the door."
    return path_to_transcript_dict


def build_path_to_transcript_dict_ESDS():
    root = "data/corpora/Emotional_Speech_Dataset_Singapore"
    path_to_transcript_dict = dict()
    for speaker_dir in os.listdir(root):
        if speaker_dir.startswith("00"):
            if int(speaker_dir) > 10:
                with open(f"{root}/{speaker_dir}/fixed_unicode.txt", mode="r", encoding="utf8") as f:
                    transcripts = f.read()
                for line in transcripts.replace("\n\n", "\n").replace(",", ", ").split("\n"):
                    if line.strip() != "":
                        filename, text, emo_dir = line.split("\t")
                        filename = speaker_dir + "_" + filename.split("_")[1]
                        path_to_transcript_dict[f"{root}/{speaker_dir}/{emo_dir}/{filename}.wav"] = text
    return path_to_transcript_dict


def build_path_to_transcript_dict_blizzard_2013():
    path_to_transcript = dict()
    root = "data/corpora/Blizzard2013/train/segmented/"
    with open(root + "prompts.gui", encoding="utf8") as f:
        transcriptions = f.read()
    blocks = transcriptions.split("||\n")
    for block in blocks:
        trans_lines = block.split("\n")
        if trans_lines[0].strip() != "":
            transcript = trans_lines[1].replace("@", "").replace("#", ",").replace("|", "").replace(";", ",").replace(
                ":", ",").replace(" 's", "'s").replace(", ,", ",").replace("  ", " ").replace(" ,", ",").replace(" .",
                                                                                                                 ".").replace(
                " ?", "?").replace(" !", "!").rstrip(" ,")
            path_to_transcript[root + "wavn/" + trans_lines[0] + ".wav"] = transcript
    return path_to_transcript


def build_file_list_singing_voice_audio_database():
    root = "data/corpora/singing_voice_audio_dataset/monophonic"
    file_list = list()
    for corw in os.listdir(root):
        for singer in os.listdir(os.path.join(root, corw)):
            for audio in os.listdir(os.path.join(root, corw, singer)):
                file_list.append(os.path.join(root, corw, singer, audio))
    return file_list


def build_path_to_transcript_dict_blizzard2023_ad():
    root = "data/corpora/Blizzard2023/AD"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace("»", '"').replace("« ", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_ad_silence_removed():
    root = "data/corpora/Blizzard2023/AD_silence_removed"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_ad_long():
    root = "data/corpora/Blizzard2023/15sec/output/AD"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_ad_long_silence_removed():
    root = "data/corpora/Blizzard2023/ad_long_silence_removed"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb():
    root = "data/corpora/Blizzard2023/NEB"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb_silence_removed():
    root = "data/corpora/Blizzard2023/NEB_silence_removed"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb_e():
    root = "data/corpora/Blizzard2023/enhanced_NEB_subset"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb_e_silence_removed():
    root = "data/corpora/Blizzard2023/enhanced_NEB_subset_silence_removed"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb_long():
    root = "data/corpora/Blizzard2023/15sec/output/NEB"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb_long_silence_removed():
    root = "data/corpora/Blizzard2023/neb_long_silence_removed"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
    return path_to_transcript


def build_path_to_transcript_dict_blizzard2023_neb_tiny_test():
    root = "data/corpora/Blizzard2023/NEB"
    path_to_transcript = dict()
    with open(os.path.join(root, "transcript.tsv"), "r", encoding="utf8") as file:
        lookup = file.read()
    for line in lookup.split("\n"):
        if line.strip() != "":
            norm_transcript = line.split("\t")[1]
            wav_path = os.path.join(root, line.split("\t")[0].split("/")[-1])
            if os.path.exists(wav_path):
                path_to_transcript[wav_path] = norm_transcript.replace("§", "").replace("#", "").replace("~", "").replace(" »", '"').replace("« ", '"').replace("»", '"').replace("«", '"')
            if len(path_to_transcript.keys()) > 50:
                break
    return path_to_transcript


def build_path_to_transcript_dict_synpaflex_norm_subset():
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "data/corpora/synpaflex-corpus/5/v0.1/"
    path_to_transcript = dict()
    for text_path in glob.iglob(os.path.join(root, "**/*_norm.txt"), recursive=True):
        with open(text_path, "r", encoding="utf8") as file:
            norm_transcript = file.read()
        path_obj = Path(text_path)
        wav_path = str((path_obj.parent.parent / path_obj.name[:-9]).with_suffix(".wav"))
        if Path(wav_path).exists():
            path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_synpaflex_all():
    """
    Contributed by https://github.com/tomschelsen
    modified by Flux to capture all audios
    """
    root = "data/corpora/synpaflex-corpus/5/v0.1/"
    path_to_transcript = dict()
    for wav_path in glob.iglob(os.path.join(root, "**/*.wav"), recursive=True):
        # two cases: either the txt lie in the txt subdir, or they lie in the txt subdir of the parent node
        file_id = str(wav_path).split("/")[-1].rstrip(".wav")
        wav_dir = "/".join(str(wav_path).split("/")[:-1])
        wav_parent_dir = "/".join(str(wav_path).split("/")[:-2])

        text_path = wav_dir + "/txt/" + file_id + "_norm.txt"
        if not Path(text_path).exists():
            text_path = wav_dir + "/txt/" + file_id + ".txt"
            if not Path(text_path).exists():
                text_path = wav_parent_dir + "/txt/" + file_id + ".wav"

        if Path(text_path).exists():
            with open(text_path, "r", encoding="utf8") as file:
                norm_transcript = file.read()
        path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


def build_path_to_transcript_dict_siwis_subset():
    """
    Contributed by https://github.com/tomschelsen
    """
    root = "data/corpora/SiwisFrenchSpeechSynthesisDatabase/"
    # part4 and part5 are not segmented
    sub_dirs = ["part1", "part2", "part3"]
    path_to_transcript = dict()
    for sd in sub_dirs:
        for text_path in glob.iglob(os.path.join(root, "text", sd, "*.txt")):
            with open(text_path, "r", encoding="utf8") as file:
                norm_transcript = file.read()
            path_obj = Path(text_path)
            wav_path = str((path_obj.parent.parent.parent / "wavs" / sd / path_obj.stem).with_suffix(".wav"))
            if Path(wav_path).exists():
                path_to_transcript[wav_path] = norm_transcript
    return path_to_transcript


if __name__ == '__main__':
    ptt = build_path_to_transcript_dict_synpaflex_all()
    print(ptt)

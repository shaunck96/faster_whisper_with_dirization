import os
import shutil
import re
import logging
from typing import List, Tuple, Dict, Union
import torch
from pydub import AudioSegment
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from whisperx.alignment import load_align_model, align
from whisperx.utils import filter_missing_timestamps
from utilities import (
    transcribe_batched,
    create_config,
    get_words_speaker_mapping,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    write_srt,
    cleanup,
)


def process_audio(audio_path: str, vocal_target: str, whisper_model_name: str, 
                  language: str, suppress_numerals: bool, batch_size: int) -> None:
    """
    Process the audio file to generate speaker-aware transcripts with proper punctuation.

    Args:
        audio_path (str): Path to the input audio file.
        vocal_target (str): Target vocal file path for processing.
        whisper_model_name (str): Name of the Whisper model to use for transcription.
        language (str): Language of the audio.
        suppress_numerals (bool): Whether to suppress numerals in transcription.
        batch_size (int): Batch size for processing.

    Returns:
        None
    """
    # Transcribe audio
    whisper_results, language = transcribe_batched(
        vocal_target,
        language,
        batch_size,
        whisper_model_name,
        "float32",
        suppress_numerals,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    # Align transcriptions
    alignment_model, metadata = load_align_model(language, "cuda")
    result_aligned = align(
        whisper_results, alignment_model, metadata, vocal_target, "cuda"
    )
    word_timestamps = filter_missing_timestamps(
        result_aligned["word_segments"],
        initial_timestamp=whisper_results[0].get("start"),
        final_timestamp=whisper_results[-1].get("end"),
    )

    # Convert audio to mono
    sound = AudioSegment.from_file(vocal_target).set_channels(1)
    temp_path = os.path.join(os.getcwd(), "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

    # Diarize speech segments
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")
    msdd_model.diarize()

    # Parse word-level speaker timestamps
    speaker_ts: List[List[Union[int, float]]] = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    # Map words to speakers
    word_speaker_mapping = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    # Restore punctuations and realign
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    words_list = [word_dict["word"] for word_dict in word_speaker_mapping]
    labeled_words = punct_model.predict(words_list)

    for word_dict, labeled_tuple in zip(word_speaker_mapping, labeled_words):
        word = word_dict["word"]
        if word and labeled_tuple[1] in ".?!":
            if word[-1] not in ",;:!" and not re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", word):
                word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

    realigned_word_speaker_mapping = get_realigned_ws_mapping_with_punctuation(
        word_speaker_mapping
    )
    sentences_speaker_mapping = get_sentences_speaker_mapping(
        realigned_word_speaker_mapping, speaker_ts
    )

    # Save transcripts
    with open(f"{os.path.splitext(audio_path)[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(sentences_speaker_mapping, f)

    with open(f"{os.path.splitext(audio_path)[0]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(sentences_speaker_mapping, srt)

    # Cleanup
    cleanup(temp_path)


if __name__ == "__main__":
    audio_path = "/content/6447183 (1).mp3"
    vocal_target = "/content/6447183 (1).mp3"
    whisper_model_name = "large-v2"
    language = "en"
    suppress_numerals = True
    batch_size = 8

    process_audio(
        audio_path,
        vocal_target,
        whisper_model_name,
        language,
        suppress_numerals,
        batch_size,
    )

import random
import os
import shutil
import re


def extract_audio_id(filename):
    """Extract the numeric audio ID from the filename."""
    match = re.search(r"audio_(\d+)_", filename)
    return int(match.group(1)) if match else float("inf")


def sample_audio_and_transcriptions(sample_size=250, new_dir="wavs_sampled/universal"):
    """Randomly samples unique audio files and their transcriptions, and creates a new transcription file."""

    # Read and parse the transcription file
    transcription_file = os.path.join("wavs/universal3", "wavs_transcriptions.txt")
    with open(transcription_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split("|") for line in f]

    # Read existing sampled transcriptions if the file exists
    existing_transcription_file = os.path.join(new_dir, "wavs_transcriptions.txt")
    if os.path.exists(existing_transcription_file):
        with open(existing_transcription_file, "r", encoding="utf-8") as f:
            existing_lines = [line.strip().split("|")[0] for line in f]
    else:
        existing_lines = []

    # Filter out already sampled lines
    lines_to_sample = [
        line for line in lines if os.path.basename(line[0]) not in existing_lines
    ]

    # Ensure the sample size does not exceed available unique samples
    sample_size = min(sample_size, len(lines_to_sample))

    # Randomly select samples
    sampled_lines = random.sample(lines_to_sample, sample_size)

    # Create the output directory if it doesn't exist
    output_dir = os.path.join("./", new_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Copy audio files and prepare new transcriptions
    new_transcriptions = []
    for audio_path, transcription in sampled_lines:
        audio_filename = os.path.basename(audio_path)
        new_audio_path = os.path.join(output_dir, audio_filename)

        # Copy the audio file to the new directory
        shutil.copy(os.path.join("./", audio_path), new_audio_path)

        # Prepare the new transcription entry
        new_transcriptions.append(
            f"wavs_sampled/universal/{audio_filename}|{transcription}"
        )

    # Read existing transcriptions and add the new ones
    if os.path.exists(existing_transcription_file):
        with open(existing_transcription_file, "r", encoding="utf-8") as f:
            existing_transcriptions = [line.strip() for line in f]
        new_transcriptions = existing_transcriptions + new_transcriptions

    # Sort the new transcriptions by natural order of audio ID
    new_transcriptions.sort(key=lambda x: extract_audio_id(x.split("|")[0]))

    # Write the new transcription file
    with open(existing_transcription_file, "w", encoding="utf-8") as f:
        for line in new_transcriptions:
            f.write(line + "\n")


def split_transcriptions(
    transcription_file="wavs_sampled/universal/wavs_transcriptions.txt",
    train_file="wavs_sampled/universal/train.txt",
    valid_file="wavs_sampled/universal/valid.txt",
    train_ratio=0.8,
):
    """Split the transcriptions into train and validation sets."""

    # Read and parse the transcription file
    with open(transcription_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    # Shuffle the lines to ensure randomness
    random.shuffle(lines)

    # Calculate split index
    split_index = int(len(lines) * train_ratio)

    # Split the lines into train and validation sets
    train_lines = lines[:split_index]
    valid_lines = lines[split_index:]

    # Write train set to train.txt
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    # Write validation set to valid.txt
    with open(valid_file, "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")


# sample_audio_and_transcriptions()

split_transcriptions()

import re
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from underthesea import text_normalize


def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        "{": "(",
        "}": ")",
        "[": "(",
        "]": ")",
        "`": "'",
        "—": "-",
        "ʼ": "'",
    }
    replace = re.compile(
        "|".join(
            [
                re.escape(k)
                for k in sorted(replacement_punctuation, key=len, reverse=True)
            ]
        ),
        flags=re.DOTALL,
    )
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

    extraneous = re.compile(r"[@#%_=\$\^&\*\+\\]")
    word = extraneous.sub("", word)
    return word


def text_cleaners(text):
    # Normalize Vietnamese text
    text = text_normalize(text)
    # Convert text to lowercase
    text = text.lower()
    # Replace multiple whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove quotation marks
    text = text.replace('"', "")
    return text


class VoiceBpeTokenizer:
    def __init__(self, vocab_file):
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt):
        txt = text_cleaners(txt)
        txt = remove_extraneous_punctuation(txt)
        return txt

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt


def preprocess_word(word, report=False):
    word = text_cleaners(word)
    word = remove_extraneous_punctuation(word)
    allowed_characters_re = re.compile(
        r'^[a-z0-9àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽềểễệỉịọỏốồổỗộớờởỡợụủứừửữựyýỳỵỹĩỷêềếệễơấầẫậấ!:;"/, \-\(\)\.\'\?ʼ]+$'
    )
    if not bool(allowed_characters_re.match(word)):
        if report and word:
            print(f"REPORTING: '{word}'")
        return ""
    return word


def batch_iterator(ttsd, batch_size=100000):
    print("Processing ASR texts.")
    for i in range(0, len(ttsd), batch_size):
        yield [preprocess_word(t, True) for t in ttsd[i : i + batch_size]]


def build_text_file_from_priors(priors, output):
    from data.audio.paired_voice_audio_dataset import (
        load_mozilla_cv,
        load_voxpopuli,
        load_tsv,
    )
    from models.audio.tts.tacotron2 import load_filepaths_and_text

    with open(output, "w", encoding="utf-8") as out:
        for p, fm in priors:
            if fm == "lj" or fm == "libritts":
                fetcher_fn = load_filepaths_and_text
            elif fm == "tsv":
                fetcher_fn = load_tsv
            elif fm == "mozilla_cv":
                fetcher_fn = load_mozilla_cv
            elif fm == "voxpopuli":
                fetcher_fn = load_voxpopuli
            else:
                raise NotImplementedError()
            apt = fetcher_fn(p)
            for path, text in apt:
                out.write(text + "\n")
            out.flush()


def train():
    with open("transcriptions.txt", "r", encoding="utf-8") as at:
        ttsd = at.readlines()

    allowed_characters_re = re.compile(
        r'^[a-z0-9àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽềểễệỉịọỏốồổỗộớờởỡợụủứừửữựyýỳỵỹĩỷêềếệễơấầẫậấ!:;"/, \-\(\)\.\'\?ʼ]+$'
    )

    def preprocess_word(word, report=False):
        word = text_cleaners(word)
        word = remove_extraneous_punctuation(word)
        if not bool(allowed_characters_re.match(word)):
            if report and word:
                print(f"REPORTING: '{word}'")
            return ""
        return word

    def batch_iterator(batch_size=1000):
        print("Processing ASR texts.")
        for i in range(0, len(ttsd), batch_size):
            yield [preprocess_word(t, True) for t in ttsd[i : i + batch_size]]

    trainer = BpeTrainer(special_tokens=["[STOP]", "[UNK]", "[SPACE]"], vocab_size=256)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ttsd))

    print(
        tokenizer.decode(
            tokenizer.encode(
                "tôi đang đi quẩy ở trongdhwq rừng đóq213 1235375t137{{}}"
            ).ids
        )
    )

    tokenizer.save("custom_vietnamese_tokenizer.json")


def test():
    tok = VoiceBpeTokenizer("custom_vietnamese_tokenizer.json")
    with open("transcriptions.txt", "r", encoding="utf-8") as at:
        ttsd = at.readlines()
        for line in ttsd:
            line = line.strip()
            seq = tok.encode(line)
            out = tok.decode(seq)
            print(f">>>{line}")
            print(f"<<<{out}")


if __name__ == "__main__":
    """
    Uncomment the following lines to build the text file from priors and train the tokenizer
    build_text_file_from_priors([('path_to_dataset1', 'format1'),
                                 ('path_to_dataset2', 'format2')],
                                 'all_texts.txt')
    """
    # train()
    test()

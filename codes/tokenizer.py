import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
from underthesea import text_normalize, sent_tokenize


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
        "|".join([re.escape(k) for k in replacement_punctuation]), flags=re.DOTALL
    )
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)
    extraneous = re.compile(r"[@#%_=\$\^&\*\+\\]")
    word = extraneous.sub("", word)
    return word


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


def batch_iterator(ttsd, batch_size=1000):
    print("Processing ASR texts.")
    for i in range(0, len(ttsd), batch_size):
        yield [preprocess_word(t, True) for t in ttsd[i : i + batch_size]]


def train_tokenizer(
    input_path,
    tokenizer_path,
    special_tokens=["[STOP]", "[UNK]", "[SPACE]"],
    vocab_size=256,
):
    with open(input_path, "r", encoding="utf-8") as at:
        # Read the file content as a single string
        content = at.read()
        # Split the content into sentences using Underthesea
        sentences = sent_tokenize(content)
        # Further split sentences into words
        ttsd = []
        for sentence in sentences:
            ttsd.extend(sentence.split())

    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(batch_iterator(ttsd), trainer, length=len(ttsd))

    tokenizer.save(tokenizer_path)

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    tokenizer_json["model"]["language"] = "vi"
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_path = "../dataset_sample/wavs_transcriptions.txt"
    tokenizer_path = "../custom_vietnamese_tokenizer.json"
    train_tokenizer(input_path, tokenizer_path)

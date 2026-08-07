"""Shared frequency-based word tokenisation for the transformer assignment.

Provides tokenise_words(), build_vocabulary(), compute_oov_rate(), and tokenise_function()
used across the notebook versions that build a custom word-frequency vocabulary instead of
a pretrained WordPiece tokeniser.
"""

import re
from collections import Counter


# define simple word-level tokenisation (replaces the WordPiece tokeniser) - add /br
def tokenise_words(text):
    """Tokenise raw review text into lowercase words.

    Strips IMDb's literal ``<br />`` HTML line-break tags and splits the
    remaining text into lowercase word tokens, ignoring punctuation.

    Args:
        text (str): Raw review text.

    Returns:
        list[str]: Lowercase word tokens extracted from the text.
    """
    text = re.sub(r"<br\s*/?>", " ", text) # remove IMDB's literal <br /> HTML line-break artifacts
    return re.findall(r"\b\w+\b", text.lower()) # split into lowercase words, ignoring punctuation


def build_vocabulary(dataset, vocab_size=20000):
    """
    Count word frequency across the datapoints and keep the top vocab_size most frequent words.

    Args:
        dataset: the dataset (IMDb).
        vocab_size (int): Maximum number of words to keep

    Returns:
        dict: Mapping from word string to integer token id.
    """
    word_counts = Counter()
    for text in dataset["text"]:
        word_counts.update(tokenise_words(text)) # accumulate word frequency across the whole corpus

    most_common_words = word_counts.most_common(vocab_size - 2) # reserve 2 slots for [PAD], [UNK]

    word_to_id = {"[PAD]": 0, "[UNK]": 1} # special tokens always occupy id 0 and 1
    for word, _ in most_common_words:
        word_to_id[word] = len(word_to_id)

    return word_to_id


def compute_oov_rate(dataset, word_to_id):
    """Compute the out-of-vocabulary (OOV) rate.

    Args:
        dataset: The dataset (IMDb split) containing a "text" column. word_to_id (dict): Mapping from word string to integer token id.

    Returns:
        float: Fraction of tokenised words not present in word_to_id.
    """
    unk_count = 0
    total_count = 0

    for text in dataset["text"]:
        words = tokenise_words(text)
        total_count += len(words)
        unk_count += sum(1 for word in words if word not in word_to_id)

    return unk_count / total_count


def tokenise_function(examples, word_to_id, unk_token_id, pad_token_id, max_seq_length):
    """Convert a batch of raw review texts into padded token id sequences.

    Tokenises each text with tokenise_words, maps words to ids via word_to_id (unknown words
    map to unk_token_id), then truncates/pads every sequence to max_seq_length.

    Args:
        examples (dict): Batch of dataset examples containing a "text" key.
        word_to_id (dict): Mapping from word string to integer token id.
        unk_token_id (int): Token id used for words not present in word_to_id.
        pad_token_id (int): Token id used to pad sequences shorter than max_seq_length.
        max_seq_length (int): Length every sequence is truncated/padded to.

    Returns:
        dict: Contains "input_ids" (list[list[int]]) and "attention_mask"
            (list[list[int]]) for the batch.
    """
    input_ids_batch = []
    attention_mask_batch = []

    for text in examples["text"]:
        words = tokenise_words(text)
        ids = [word_to_id.get(word, unk_token_id) for word in words] # unknown words map to [UNK]

        ids = ids[:max_seq_length] # truncate to max_seq_length
        attention_mask = [1] * len(ids) # 1 marks a real token

        padding_length = max_seq_length - len(ids)
        ids = ids + [pad_token_id] * padding_length # pad up to max_seq_length
        attention_mask = attention_mask + [0] * padding_length # 0 marks a padding position

        input_ids_batch.append(ids)
        attention_mask_batch.append(attention_mask)

    return {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

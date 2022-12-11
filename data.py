from enum import Enum
from pathlib import Path

from tokenizers import Tokenizer
# источник: https://github.com/huggingface/tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch

from torch.utils.data import Dataset


class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"


def process_training_file(input_path: Path, output_path: Path):
    """
    Processes raw training files ("train.tags.SRC-TGT.*"), saving
    the output as a sequence of unformatted examples
    (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    lines = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if (line[:1] == '<') or (line[:2] == ' <'):
                continue
            lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines([line + '\n' for line in lines])

    return lines


def process_evaluation_file(input_path: Path, output_path: Path):
    """
    Processes raw validation and testing files ("IWSLT17.TED.{dev,test}2010.SRC-TGT.*.xml"),
    saving the output as a sequence of unformatted examples (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    lines = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if (line[:4] == '<seg') or (line[:5] == ' <seg'):
                lines.append(line[line.find('>') + 1: line.rfind('<')])

    with open(output_path, 'w') as f:
        f.writelines([line + '\n' for line in lines])

    return lines


def convert_files(base_path: Path, output_path: Path):
    """
    Given a directory containing all the dataset files, convert
    each one into the "one example per line" format.
    :param base_path: Path containing files with original data
    :param output_path: Path containing files with processed data
    """

    for language in ["de", "en"]:
        process_training_file(
            base_path / f"train.tags.de-en.{language}",
            output_path / f"train.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.dev2010.de-en.{language}.xml",
            output_path / f"val.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.tst2010.de-en.{language}.xml",
            output_path / f"test.{language}.txt",
        )


def train_tokenizers(base_dir: Path, save_dir: Path):
    """
    Trains tokenizers for source and target languages and saves them to `save_dir`.
    :param base_dir: Directory containing processed training and validation data
    (.txt files from `convert_files`)
    :param save_dir: Directory for storing trained tokenizer data
    (two files: `tokenizer_de.json` and `tokenizer_en.json`)
    """
    from tokenizers.processors import TemplateProcessing
    from tokenizers import normalizers
    from tokenizers.normalizers import NFD
    from tokenizers.normalizers import Lowercase

    for language in ['de', 'en']:
        # источник: https://github.com/huggingface/tokenizers
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase()])
        tokenizer.post_processor = TemplateProcessing(
            single=f"[BOS] $A [EOS]", special_tokens=[("[BOS]", 0), ("[EOS]", 1)]
        )

        trainer = BpeTrainer(vocab_size=30000,
                             show_progress=True,
                             special_tokens=["[BOS]", "[EOS]", "[PAD]", "[UNK]"])

        # https://huggingface.co/docs/tokenizers/quicktour
        fs = [str(base_dir.joinpath(f"{regime}.{language}.txt")) for regime in ['train', 'val']]
        tokenizer.train(files=fs, trainer=trainer)

        tokenizer.save(str(save_dir.joinpath(f"{language}_tokenizer.json")))


class TranslationDataset(Dataset):
    def __init__(
            self,
            src_file_path,
            tgt_file_path,
            src_tokenizer: Tokenizer,
            tgt_tokenizer: Tokenizer,
            max_len=32,
    ):
        """
        Loads the training dataset and parses it into separate tokenized training examples.
        No padding should be applied at this stage
        :param src_file_path: Path to the source language training data
        :param tgt_file_path: Path to the target language training data
        :param src_tokenizer: Trained tokenizer for the source language
        :param tgt_tokenizer: Trained tokenizer for the target language
        :param max_len: Maximum length of source and target sentences for each example:
        if either of the parts contains more tokens, it needs to be filtered.
        """
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path

        self.src_tokenizer = src_tokenizer
        self.tknz_src_texts = []

        self.tgt_tokenizer = tgt_tokenizer
        self.tknz_target_texts = []

        with open(self.src_file_path, 'r') as f_in:
            with open(self.tgt_file_path, 'r') as f_out:
                for line_in, line_out in zip(f_in, f_out):
                    tok_in = self.src_tokenizer.encode(line_in)
                    tok_out = self.tgt_tokenizer.encode(line_out)
                    if len(tok_in) > max_len or len(tok_out) > max_len:
                        continue
                    self.tknz_src_texts += [tok_in.ids]
                    self.tknz_target_texts += [tok_out.ids]

    def __len__(self):
        return len(self.tknz_src_texts)

    def __getitem__(self, i):
        return {
            'source': self.tknz_src_texts[i],
            'target': self.tknz_target_texts[i]
        }

    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into
        `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader
        class for training and validation datasets in your pipeline.
        """
        sources, targets = [], []
        paddings_src, paddings_trgt = [], []
        s_max_len, t_max_len = None, None
        for text_pair in batch:
            if s_max_len is None:
                s_max_len = len(text_pair['source'])
                t_max_len = len(text_pair['target'])
            num_pad_src = s_max_len - len(text_pair['source'])
            num_pad_trg = t_max_len - len(text_pair['target'])
            if len(text_pair['source']) > s_max_len:
                for s, p in zip(sources, paddings_src):
                    s += [2] * (- num_pad_src)
                    p += [1] * (- num_pad_src)

            if len(text_pair['target']) > t_max_len:
                for t, p in zip(targets, paddings_trgt):
                    t += [2] * (- num_pad_trg)
                    p += [1] * (- num_pad_trg)
            sources.append(text_pair['source'])
            targets.append(text_pair['target'])
            paddings_src.append([0] * len(text_pair['source']))
            paddings_trgt.append([0] * len(text_pair['target']))

            if num_pad_src > 0:
                sources[-1] += [2] * num_pad_src
                paddings_src[-1] += [1] * num_pad_src

            if num_pad_trg > 0:
                targets[-1] += [2] * num_pad_trg
                paddings_trgt[-1] += [1] * num_pad_trg

            s_max_len = max(s_max_len, len(text_pair['source'])) if s_max_len else len(text_pair['source'])
            t_max_len = max(t_max_len, len(text_pair['target'])) if t_max_len else len(text_pair['target'])

        target_seq_len = len(paddings_trgt[0])
        memory = torch.triu(
            torch.ones(target_seq_len - 1, target_seq_len - 1), diagonal=1)

        # print([len(s) for s in sources])
        # print([len(t) for t in targets])
        # print(torch.tensor(paddings_src, dtype=torch.bool).shape)
        # print(torch.tensor(paddings_trgt, dtype=torch.bool).shape)

        collated_result = {
            'source': torch.tensor(sources, dtype=torch.long),
            'paddings_src': torch.tensor(paddings_src, dtype=torch.bool),
            'target': torch.tensor(targets, dtype=torch.long),
            'padding_target': torch.tensor(paddings_trgt, dtype=torch.bool),
            'memory_target': memory.type(torch.bool)
        }

        return collated_result




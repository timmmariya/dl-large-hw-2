import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tokenizers import Tokenizer

from model import TranslationModel
from torch.utils.data import DataLoader
import numpy as np

# it's a surprise tool that will help you later
detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()


# забыла определить два этих метода при создании класса модели
# уже обучила без них, не ругайте за такой костыль, пожалуйста:)
def model_encode(model: TranslationModel,
                 src: torch.Tensor):
    return model.transformer.encoder(
        model.positional_encoding(model.src_tok_emb(src)))


def model_decode(model: TranslationModel,
                 tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
    return model.transformer.decoder(
        model.positional_encoding(model.tgt_tok_emb(tgt)), memory, tgt_mask)


@torch.no_grad()
def _greedy_decode(
        model: TranslationModel,
        src: torch.Tensor,
        max_len: int,
        device: torch.device,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with greedy search.
    The decoding procedure terminates once either max_len steps have passed
    or the "end of sequence" token has been reached for all sentences in the batch.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param device: device that the model runs on
    :return: a (batch, time) tensor with predictions
    """
    # already (batch_size, source sentence tokens)
    source = src.to(device)
    memory = model_encode(model, source).to(device)

    preds = torch.zeros((src.size(0), 1), dtype=torch.long).to(device)  # fill with the BOS token id

    for i in range(max_len - 1):
        tgt_mask = torch.triu(torch.ones((preds.size(1), preds.size(1))), diagonal=1)
        tgt_mask = tgt_mask.type(torch.bool)

        out = model_decode(model, preds, memory, tgt_mask)
        probs = model.fc(out[:, -1, :])

        new_word = torch.argmax(probs, dim=-1).unsqueeze(1)
        preds = torch.cat([preds, new_word], dim=1)

        if torch.all(new_word == 1):
            break
    return preds


def _beam_search_decode(
        model: TranslationModel,
        scr: torch.Tensor,
        max_len: int,
        device: torch.device,
        beam_size: int,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with beam search.
    The decoding procedure terminates once max_len steps have passed.
    :param model: the model to use for translation
    :param scr_batch: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param device: device that the model runs on
    :param beam_size: the number of hypotheses
    :return: a (batch, time) tensor with predictions
    """
    source = src.to(device)
    memory = model_encode(model, source).to(device)

    preds = torch.zeros((src.size(0), 1), dtype=torch.long).to(device)  # fill with the BOS token id

    tgt_mask = torch.triu(torch.ones((preds.size(1), preds.size(1))), diagonal=1)
    tgt_mask = tgt_mask.type(torch.bool).to(device)

    out = model_decode(model, preds, memory, tgt_mask)
    logits = nn.LogSoftmax(dim=-1)(model.fc(out[:, -1]))

    next_probs, tokens = logits.topk(beam_size, dim=-1)
    next_probs = next_probs.flatten()
    tokens = tokens.flatten()

    preds = torch.cat([preds.flatten().repeat(beam_size).reshape(-1, 1),
                       tokens.reshape(-1, 1)], dim=-1)

    matrix = [[[i] * beam_size] for i in len(memory)]
    memory = memory[torch.tensor(matrix).flatten()]

    # ...

    return preds


@torch.inference_mode()
def translate(
        model: torch.nn.Module,
        test_loader: DataLoader,
        tgt_tokenizer: Tokenizer,
        translation_mode: str,
        device: torch.device,
):
    """
    Given a list of sentences, generate their translations.
    :param model: the model to use for translation
    :param test_loader: untokenized source sentences
    :param tgt_tokenizer: target language tokenizer
    :param translation_mode: either "greedy", "beam" or anything more advanced
    :param device: device that the model runs on
    """
    final = []
    from data import SpecialTokens

    i = 1
    for batch in test_loader:
        if translation_mode == 'greedy':
            res = _greedy_decode(model, batch['source'], 300, device)
        elif translation_mode == 'beam':
            res = _beam_search_decode(model, batch['source'], 300, device, 5)
            pass

        for ans in res.cpu().numpy():
            if len(np.where(ans == 1)[0]) > 0:
                crop_from = np.min(np.where(ans == 1)[0])
            else:
                crop_from = len(ans)
            final += [tgt_tokenizer.decode(ans[:crop_from])]
        i += 1

    print(f'{i} batches proсcessed')
    return final





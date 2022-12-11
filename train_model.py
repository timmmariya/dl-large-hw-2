from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import torch
from sacrebleu.metrics import BLEU
from tokenizers import Tokenizer
from tqdm import trange
import wandb

from data import TranslationDataset
# from decoding import translate
from model import TranslationModel
from torch.utils.data import DataLoader


def train_epoch(model, optimizer, scheduler, train_dataloader,
                criterion, device=torch.device('cuda:0')):
    # train the model for one epoch
    # you can obviously add new arguments or change the API if it does not suit you
    model.train()
    losses = 0

    for scr_trg_dict in tqdm(train_dataloader):
        #         wandb.log({'learning rate': scheduler.get_last_lr()[-1]})

        src = scr_trg_dict['source'].to(device)
        src_padding_mask = scr_trg_dict['paddings_src'].to(device)

        tgt = scr_trg_dict['target'].to(device)
        tgt_padding_mask = scr_trg_dict['padding_target'].to(device)
        tgt_mask = scr_trg_dict['memory_target'].to(device)

        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input, tgt_mask,
                       src_padding_mask, tgt_padding_mask[:, :-1])

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:]

        loss = criterion(logits.transpose(-1, -2), tgt_out)
        loss.backward()

        optimizer.step()
        scheduler.step()

        wandb.log({'train_loss': loss.item()})
        losses += loss.item()

    return losses / len(train_dataloader)


@torch.inference_mode()
def evaluate(model, val_dataloader, criterion, device=torch.device('cuda:0')):
    # compute the loss over the entire validation subset
    model.eval()
    losses = 0

    for scr_trg_dict in tqdm(val_dataloader):
        src = scr_trg_dict['source'].to(device)
        src_padding_mask = scr_trg_dict['paddings_src'].to(device)

        tgt = scr_trg_dict['target'].to(device)
        tgt_padding_mask = scr_trg_dict['padding_target'].to(device)
        tgt_mask = scr_trg_dict['memory_target'].to(device)

        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input, tgt_mask,
                       src_padding_mask, tgt_padding_mask[:, :-1])

        tgt_out = tgt[:, 1:]
        loss = criterion(logits.transpose(-1, -2), tgt_out)
        losses += loss.item()

    return losses / len(val_dataloader)


def train_model(data_dir, tokenizer_path, num_epochs):
    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "de_tokenizer.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "en_tokenizer.json"))

    train_dataset = TranslationDataset(
        data_dir / "train.de.txt",
        data_dir / "train.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,  # might be enough at first
    )
    val_dataset = TranslationDataset(
        data_dir / "val.de.txt",
        data_dir / "val.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128,
    )

    d = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(d)
    torch.manual_seed(222)

    model = TranslationModel(
        num_encoder_layers=6,
        num_decoder_layers=6,
        emb_size=256,
        nhead=8,
        src_vocab_size=30_000,
        tgt_vocab_size=30_000,
    )

    print('Model has been initialized... device: ', d)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=8,
                                  collate_fn=train_dataset.collate_translation_data)
    val_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=8,
                                collate_fn=val_dataset.collate_translation_data)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_dataloader),
                                                    max_lr=1e-3, pct_start=0.2, epochs=num_epochs)

    wandb.init(project="hw2-Timonina", name='fourth_trial')

    # create loss, optimizer, scheduler objects, dataloaders etc.
    # don't forget about collate_fn
    # if you intend to use AMP, you might need something else

    min_val_loss = float("inf")

    # ACTUAL TRAINING
    for epoch in trange(1, num_epochs + 1):

        train_loss = train_epoch(model, optimizer, scheduler, train_dataloader, criterion, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        wandb.log({'val_loss': val_loss})

        # might be useful to translate some sentences from validation to check your decoding implementation

        # also, save the best checkpoint somewhere around here
        if val_loss < min_val_loss:
            print("New best loss! Saving checkpoint")
            torch.save(model.state_dict(), "checkpoint_best.pth")
            min_val_loss = val_loss

        # and the last one in case you need to recover
        # by the way, is this sufficient?
        torch.save(model.state_dict(), "checkpoint_last.pth")

    # load the best checkpoint
    model.load_state_dict(torch.load("checkpoint_best.pth"))
    return model


def translate_test_set(model: TranslationModel, data_dir, tokenizer_path):
    d = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(d)
    torch.manual_seed(222)

    model.eval()

    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "de_tokenizer.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "en_tokenizer.json"))

    test_dataset = TranslationDataset(
        data_dir / "test.de.txt",
        data_dir / "test.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=128
    )

    from decoding import translate

    # translate with greedy search
    greedy_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
            "answers_greedy.txt", "w+") as output_file:
        test_loader = DataLoader(test_dataset, batch_size=128,
                                 collate_fn=test_dataset.collate_translation_data,
                                 shuffle=False, num_workers=8)
        greedy_translations = translate(model,
                                        test_loader,
                                        tgt_tokenizer,
                                        'greedy', device)
        output_file.write('\n'.join(greedy_translations))

    beam_translations = []
    with open(data_dir / "test.de.txt") as input_file, open("answers_beam.txt", "w+") as output_file:
        test_loader = DataLoader(test_dataset, batch_size=128,
                                 collate_fn=test_dataset.collate_translation_data,
                                 shuffle=False, num_workers=8)
        beam_translations = translate(model,
                                        test_loader,
                                        tgt_tokenizer,
                                        'beam', device)
        output_file.write('\n'.join(beam_translations))

    with open(data_dir / "test.en.txt") as input_file:
        references = [line.strip() for line in input_file]

    bleu = BLEU()
    bleu_greedy = bleu.corpus_score(greedy_translations, [references]).score

    # we're recreating the object, as it might cache some stats
    #     bleu = BLEU()
    #     bleu_beam = bleu.corpus_score(beam_translations, [references]).score

    print(f"BLEU with greedy search: {bleu_greedy}, with beam search: bleu_beam")
    # maybe log to wandb/comet/neptune as well


if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to the directory containing processed data"
    )
    data_group.add_argument(
        "--tokenizer-path", type=Path, help="Path to the trained tokenizer files"
    )

    # argument groups are useful for separating semantically different parameters
    hparams_group = parser.add_argument_group("Training hyperparameters")
    hparams_group.add_argument(
        "--num-epochs", type=int, default=10, help="Number of training epochs"
    )

    args = parser.parse_args()

    model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs)
    #     model = TranslationModel(
    #         num_encoder_layers=6,
    #         num_decoder_layers=6,
    #         emb_size=256,
    #         nhead=8,
    #         src_vocab_size=30_000,
    #         tgt_vocab_size=30_000,
    #     )
    #     d = "cuda" if torch.cuda.is_available() else "cpu"
    #     device = torch.device(d)
    #     torch.manual_seed(222)

    #     model.to(device)
    #     model.load_state_dict(torch.load("checkpoint_best.pth"))
    translate_test_set(model, args.data_dir, args.tokenizer_path)


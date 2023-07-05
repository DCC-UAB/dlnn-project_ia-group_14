import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

from utils.utils import translate_sentence, save_checkpoint, load_checkpoint
from models.models import Encoder, Decoder, Seq2Seq

# DATA
spacy_german = spacy.load("de_core_news_sm")
spacy_english = spacy.load("en_core_web_sm")

def tokenizer_german(text):
    return [token.text for token in spacy_german.tokenizer(text)] # [::-1]

def tokenizer_english(text):
    return [token.text for token in spacy_english.tokenizer(text)]

german = Field(tokenize=tokenizer_german, lower=True,
                init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_english, lower=True,
                init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data=Multi30k.splits(
    exts=('.de', '.en'),
    fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

# TRAIN
num_epochs=20
lr=0.001
batch_size= 64

load_model=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = len(german.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 250
decoder_embedding_size = 250
hidden_size=512
num_layers=2
encoder_dropout=0.5
decoder_dropout=0.5

writer = SummaryWriter(f'runs/loss_plot')
step=0

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    device=device
)

encoder_net = Encoder(input_size, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
decoder_net = Decoder(output_size, decoder_embedding_size, hidden_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net, device, 0.5).to(device) #0.5 is teacher force ratio

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

pad_idx = english.vocab.stoi[english.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=lr)

if load_model:
    load_checkpoint(torch.load('checkpoint.pth'), model, optimizer)

test_sentence = 'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern'

for epoch in range(num_epochs):
    print(f'Epoch: {epoch} / {num_epochs}')

    model.eval()

    translated_sentence = translate_sentence(model, test_sentence, german, english, device, max_length=50)
    print(f'Test translated sentence: {translated_sentence}')

    model.train()

    for batch_index, batch in enumerate(train_iter):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        optimizer.zero_grad()

        output = model(input_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training Loss', loss, global_step=step)
        step += 1

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)
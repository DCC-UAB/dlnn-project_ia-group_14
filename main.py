import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import numpy as np
import spacy
import random

from utils.utils import translate_sentence, save_checkpoint, load_checkpoint
from models.models import Encoder, Decoder, Seq2Seq

import sys

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

load_model=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size=64

input_size = len(german.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 250
decoder_embedding_size = 250
hidden_size=512
num_layers=2
encoder_dropout=0.5
decoder_dropout=0.5
lr=0.001

step=0

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    device=device
)

encoder_net = Encoder(input_size, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
decoder_net = Decoder(output_size, decoder_embedding_size, hidden_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

if torch.cuda.is_available():
    load_checkpoint(torch.load('checkpoint.pth'), model, optimizer)
else:
    load_checkpoint(torch.load('checkpoint.pth', map_location=torch.device('cpu')), model, optimizer)

if __name__ == '__main__':
    print()
    model.eval()

    cmd_input = len(sys.argv) > 1
    test_sentence = 'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern'

    if cmd_input:
        test_sentence = sys.argv[1]
    else:
        print(f'Using example sentence')

    translated_sentence = ' '.join(translate_sentence(model, test_sentence, german, english, device, max_length=50))

    print()
    print(f'Input sentence:\n{test_sentence}')
    print()
    print(f'Translated sentence:\n{translated_sentence}')
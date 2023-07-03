from posixpath import split
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

# DATA
spacy_german = spacy.load("de_core_news_sm")
spacy_english = spacy.load("en_core_web_sm")

def tokenizer_german(text):
    return [token.text for token in spacy_german.tokenizer(text)]

def tokenizer_english(text):
    return [token.text for token in spacy_english.tokenizer(text)]

german = Field(tokenize=tokenizer_german, lower=True,
                init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_english, lower=True,
                init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data=Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

print(f"Unique tokens in source (de) vocabulary: {len(german.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(english.vocab)}")

# MODELS
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N)

        embedding=self.dropout(self.embedding(x))
        # embedding shape: (seq_len, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        self.dropout=nn.Dropout(p)
        self.embedding=nn.Embedding(input_size, embedding_size)
        self.rnn=nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc=nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        # x shape:(1, N)

        embedding=self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        prediction=self.fc(outputs)
        # prediction shape: (1, N, vocab_len)

        prediction.unsqueeze(0)

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size=source.shape[1]
        target_len=target.shape[0]
        
        target_vocab_size= len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # get start token
        x = target[0]

        for t in range(1, target_len):
            test = random.random()

            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
        
            best_guess=output.argmax()
            # best guess shape:(N, eng_vocab_size)

            print(f'{test} < {teacher_force_ratio} else {best_guess}')
            x = target[t] if test < teacher_force_ratio else best_guess

        return outputs

# TRAIN
num_epochs=1
lr=0.001
batch_size= 64

load_model=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_encoder_size = len(german.vocab)
input_decoder_size = len(english.vocab)

output_size = len(english.vocab)
encoder_embedding_size = 250
decoder_embedding_size = 250
hidden_size=1024
num_layers=2

encoder_dropout=0.5
decoder_dropout=0.5

writer = SummaryWriter(f'runs/loss_plot')
step=0

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device
)

encoder_net = Encoder(input_encoder_size, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)

decoder_net = Decoder(input_decoder_size, decoder_embedding_size, hidden_size, output_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=lr)

test_sentence = 'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern'

if load_model:
    load_checkpoint(torch.load('checkpoint_pth.tar'), model, optimizer)

for epoch in range(num_epochs):
    print(f'Epoch: {epoch} / {num_epochs}')

    checkpoint = {'state_dict': model.state_dict, 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(model, test_sentence, german, english, device, max_length=50)

    print(f'Test translated sentence: {translated_sentence}')

    model.train()

    for batch_index, batch in enumerate(train_iter):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(input_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training Loss', loss, global_step=step)
        step += 1


        

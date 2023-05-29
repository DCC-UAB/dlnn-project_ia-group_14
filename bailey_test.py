from turtle import forward
from xml.dom.pulldom import CHARACTERS
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

from utils.utils import EOS_token, SOS_token, encoding_characters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 100

SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size=output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)






        

hidden_size = 256
# encoder = EncoderRNN(number_of_input_characters, hidden_size)
# decoder = AttnDecoderRNN(hidden_size, number_of_output_characters)

# train(encoder, decoder, n_iters=75000, print_every=5000)


def trainingLoop(encoder, decoder, n_iters, learning_rate=0.01):
    plot_losses = []

    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    encoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    training_data = [] # this will be a object that has 2 tensors one containing the input, the other containing target texts

    for iter in range(1, n_iters + 1):
        input_text, target_text = training_data[iter - 1]

def train(input, target, encoder, decoder, encoder_optim, decoder_optim, criterion, max_length=max_length, teacher_forcing_ratio = 0.5):
    hidden = encoder.initHidden()

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    input_length = input.size(0)
    target_length = target.size(0)

    encoder_outputs = torch.zero(max_length, encoder.hidden_size, device=device)

    loss = 0

    for char in input:
        # loop over input and give each character to encoder one at a time
        encoder_output, hidden = encoder(
            input[char], hidden
        )
        encoder_outputs[char] = encoder_output[0, 0]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    decoder_input = torch.tensor([[SOS_token]], device=device)

    if use_teacher_forcing:
        for char in target:
            decoder_output, hidden, decoder_attention = decoder(
                decoder_input, hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target[char])
            decoder_input = target[char]

    else:
        for char in target:
            decoder_output, hidden, decoder_attention = decoder(
                decoder_input, hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze.detach() 

            loss += criterion(decoder_output, target[char])
            if decoder_input.item() == EOS_token:
                break







class DatasetFromTextFile():
    def __init__(self, data_path):
        self.input_characters, self.target_characters, \
        self.input_texts, self.target_texts \
            = self._extract_characters(data_path)

        self._encode_characters()



    def _extract_characters(self, data_path):
        input_texts = []
        target_texts = []
        
        input_characters = set()
        target_characters = set()

        lines = open(data_path).read().split('\n')

        print(str(len(lines) - 1))

        for line in lines[: len(lines) -1]:
            input_text, target_text = line.split('\t')[:2]
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)

            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

        return input_characters, target_characters, input_texts, target_texts

    def _encode_characters(self):
        # We need to encode the dataset using one hot enconding

        num_encoder_tokens = len(self.input_characters) # total number of unique characters from input language
        num_decoder_tokens = len(self.target_characters) # total number of unique characters from target language

        max_encoder_seq_length = max([len(txt) for txt in self.input_texts]) # longest sentence in input sentences
        max_decoder_seq_length = max([len(txt) for txt in self.target_texts]) # longest sentence in target sentences

        #print('Number of num_encoder_tokens:', num_encoder_tokens)
        #print('Number of samples:', len(self.input_texts))
        #print('Number of unique input tokens:', num_encoder_tokens)
        #print('Number of unique output tokens:', num_decoder_tokens)
        #print('Max sequence length for inputs:', max_encoder_seq_length)
        #print('Max sequence length for outputs:', max_decoder_seq_length)

        input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])

        print(input_token_index)
        print(target_token_index)

        encoder_input_data = np.zeros((len(self.input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        decoder_input_data = np.zeros((len(self.input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        decoder_target_data = np.zeros((len(self.target_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

        #print(encoder_input_data.shape)
        #print(decoder_input_data.shape)
        #print(decoder_target_data.shape)

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.

            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[char]] = 1

                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1

        #return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length


if __name__ == '__main__':
    data_path = 'datasets/deu.txt'

    GermanData = DatasetFromTextFile(data_path)
        
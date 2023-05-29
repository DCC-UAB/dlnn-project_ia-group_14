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

#########
# MODELS

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, 300)
        self.gru = nn.GRU(300, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        print(input.shape, hidden.shape)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device)

    
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


##########
# TRAINING

def training_loop(dataset, encoder, decoder, n_iters, learning_rate=0.01):
    print_loss_total = 0

    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    input_data, target_data = dataset.get_data() # this will be a object that has 2 tensors one containing the input, the other containing target texts

    for iter in range(1, n_iters + 1): # could this be changed to range(0, n_iters)
        input_text = input_data[iter - 1]
        target_text = target_data[iter - 1]

        loss = train(input_text, target_text, encoder, decoder, encoder_optim, decoder_optim, criterion)

        print_loss_total += loss

        if iter % 1000 == 0:
            print_loss_avg = print_loss_total / 1000
            print_loss_total = 0
            print(f'iteration: {iter} / {n_iters} | loss: {print_loss_avg}') 


def train(input, target, encoder, decoder, encoder_optim, decoder_optim, criterion, teacher_forcing_ratio = 0.5):
    hidden = encoder.initHidden()

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    print(input, input.shape)

    for char in input:
        encoder_output, hidden = encoder(
            char, hidden
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


#########
# DATASET

class DatasetFromTextFile():
    '''
        This class will read the text file and extract each sentence from 
        input and target languages.

        next it encodes the data so that it can be read by the model
    '''
    def __init__(self, data_path, num_of_samples):
        self.input_characters, self.target_characters, \
        self.input_texts, self.target_texts \
            = self._extract_characters(data_path, num_of_samples)

        self.input_char_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
        self.target_char_index = dict([(char, i) for i, char in enumerate(self.target_characters)]) 

        self.num_input_chars = len(self.input_characters)
        self.num_target_chars = len(self.target_characters)
        
        self.encoded_input_data, self.decoded_input_data, self.decoded_target_data \
            = self._encode_characters()



    def _extract_characters(self, data_path, num_of_samples):
        '''
            Open the file and split lines and tabs
            store input and target texts
            record each unique character
        '''
        input_texts = []
        target_texts = []
        
        input_characters = set()
        target_characters = set()

        lines = open(data_path).read().split('\n')

        print(str(len(lines) - 1))

        for line in lines[: min(num_of_samples, len(lines) -1)]:
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
        '''
            For the model to be able to understand the data we create a dictionary
            that will store each character as a number
            e.g.
            {' ': 0, '!': 1, 'A': 2, 'B': 3, 'C': 4, ... 'z': 69}

            input tensor shape is (number of samples, input sentence(with padding), number of unique characters)
        '''
        max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        max_decoder_seq_length = max([len(txt) for txt in self.target_texts]) 

        encoded_input_data = np.zeros((len(self.input_texts), max_encoder_seq_length, self.num_input_chars), dtype='float32') # (261499, 537, 103)
        decoded_input_data = np.zeros((len(self.input_texts), max_decoder_seq_length, self.num_target_chars), dtype='float32') # (261499, 493, 126)
        decoded_target_data = np.zeros((len(self.input_texts), max_decoder_seq_length, self.num_target_chars), dtype='float32') # (261499, 493, 126)

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            print(len(input_text))
            break
            for t, char in enumerate(input_text):
                encoded_input_data[i, t, self.input_char_index[char]] = 1.

            for t, char in enumerate(target_text):
                decoded_input_data[i, t, self.target_char_index[char]] = 1

                if t > 0:
                    decoded_target_data[i, t - 1, self.target_char_index[char]] = 1

        return encoded_input_data, decoded_input_data, decoded_target_data

    def get_data(self):
        return torch.from_numpy(self.encoded_input_data).int(), torch.from_numpy(self.decoded_target_data).int()


if __name__ == '__main__':
    # Firstly we want to extract the data from the datafile
    data_path = 'datasets/deu.txt'
    german_data = DatasetFromTextFile(data_path, num_of_samples=10000)

    print(german_data.input_char_index)

    hidden_size = 256

    encoder = EncoderRNN(german_data.num_input_chars, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, german_data.num_target_chars)

    training_loop(german_data, encoder, decoder, n_iters=50000)

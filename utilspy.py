import torch
import torch.nn as nn

#some prints were just for help in the colab)

def extract_characters(data_path, exchange_language=False):
    input_texts = []
    target_texts = []
    input_chars = set()
    target_chars = set()
    lines = open(data_path).read().split('\n')
    num_samples = len(lines) - 1

    if not exchange_language:
        for line in lines[:min(num_samples, len(lines) - 1)]:
            if '\t' in line:
                parts = line.split('\t')
                input_text = parts[0]
                target_text = '\t'.join(parts[1:])
                target_text = '\t' + target_text + '\n'
                input_texts.append(input_text)
                target_texts.append(target_text)
                for char in input_text:
                    if char not in input_chars:
                        input_chars.add(char)
                for char in target_text:
                    if char not in target_chars:
                        target_chars.add(char)
            else:
                print(f"ignor line: {line} - tab character misses")

    else:
        for line in lines[:min(num_samples, len(lines) - 1)]:
            if '\t' in line:
                parts = line.split('\t')
                target_text = parts[0]
                input_text = '\t'.join(parts[1:])
                target_text = '\t' + target_text + '\n'
                input_texts.append(input_text)
                target_texts.append(target_text)
                for char in input_text:
                    if char not in input_chars:
                        input_chars.add(char)
                for char in target_text:
                    if char not in target_chars:
                        target_chars.add(char)
            else:
                print(f"ignor line: {line} - tab character misses")

    input_chars = sorted(list(input_chars))
    target_chars = sorted(list(target_chars))

    print("Extracted characters:")
    print("Input characters:", input_chars)
    print("Target characters:", target_chars)
    print()

    return input_chars, target_chars, input_texts, target_texts

#still wasn't able to run correctly the SpaCy version of the function in colab.

def prepare_data(data_path, exchange_language=False):
    input_chars, target_chars, input_texts, target_texts = extract_characters(data_path, exchange_language)
    return encode_data(input_chars, target_chars, input_texts, target_texts)

def encode_data(input_chars, target_chars, input_texts, target_texts):
    input_char_to_index = dict([(char, i) for i, char in enumerate(input_chars)])
    target_char_to_index = dict([(char, i) for i, char in enumerate(target_chars)])

    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    for i in range(len(input_texts)):
        encoder_input = [input_char_to_index[char] for char in input_texts[i]]
        encoder_input += [0] * (max_encoder_seq_length - len(encoder_input))
        encoder_input_data.append(encoder_input)

        decoder_input = [target_char_to_index[char] for char in target_texts[i][:-1]]
        decoder_input += [0] * (max_decoder_seq_length - len(decoder_input))
        decoder_input_data.append(decoder_input)

        decoder_target = [target_char_to_index[char] for char in target_texts[i][1:]]
        decoder_target += [0] * (max_decoder_seq_length - len(decoder_target))
        decoder_target_data.append(decoder_target)

    encoder_input_data = torch.tensor(encoder_input_data, dtype=torch.long)
    decoder_input_data = torch.tensor(decoder_input_data, dtype=torch.long)
    decoder_target_data = torch.tensor(decoder_target_data, dtype=torch.long)

    return encoder_input_data, decoder_input_data, decoder_target_data, input_char_to_index, target_char_to_index, max_encoder_seq_length, max_decoder_seq_length




##########
#import the necesseray modules of Pytorch, numpy and pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from collections import Counter
import time
import os
import copy
import numpy as np #added from util from starting point
import _pickle as pickle # added from the same
print("PyTorch Version: ",torch.__version__)



#this code comes from the "utils" file from the starting point of the project and modified for Pytorch library

def loadEncoderDecoderModel():
# We load the encoder model and the decoder model and their respective weights
    encoder_model= DataLoader(encoder_path)
    decoder_model= DataLoader(decoder_path)
    return encoder_model,decoder_model

def decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index):\
# We run the model and predict the translated sentence

    # We encode the input
    states_value = encoder_model.predict(input_seq)

    
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    
    target_seq[0, 0, target_token_index['\t']] = 1.


    stop_condition = False
    decoded_sentence = ''
    # We predict the output letter by letter 
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # We translate the token in hamain language
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # We check if it is the end of the string
        if (sampled_char == '\n' or
           len(decoded_sentence) > 500):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence


def encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens):
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for t, char in enumerate(sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data

def saveChar2encoding(filename,input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index):
    f = open(filename, "wb")
    pickle.dump(input_token_index, f)
    pickle.dump(max_encoder_seq_length, f)
    pickle.dump(num_encoder_tokens, f)
    pickle.dump(reverse_target_char_index, f)
    
    pickle.dump(num_decoder_tokens, f)
    
    pickle.dump(target_token_index, f)
    f.close()
    

def getChar2encoding(filename):
    f = open(filename, "rb")
    input_token_index = pickle.load(f)
    max_encoder_seq_length = pickle.load(f)
    num_encoder_tokens = pickle.load(f)
    reverse_target_char_index = pickle.load(f)
    num_decoder_tokens = pickle.load(f)
    target_token_index = pickle.load(f)
    f.close()
    return input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index




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


from curses import noraw
import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *

import unicodedata
import re
import random


SOS_token = 0 # start of string
EOS_token = 1 # end of string

class Lang():
    '''
        This is a helper sub that will keep track of unique words.
        We split every sentence into words and tack how often the words are seen and
        index each word.
    '''
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2 # count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# input data is in Unicode, we will covert to ASCII for simplicities sake
# 
# Thanks to > https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Make data lowercase, trim, and remove non-letter characters using regex
# For more reading on Regex > https://regex101.com/ || https://docs.python.org/3/library/re.html
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s) # Match a single character present in the list below [.!?] and insert a space before that character. e.g. 'Hello!' -> 'Hello !'
    # Maybe we dont want this? || s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) # Match a single character not present in the list below [^a-zA-Z.!?] and replace it with ' '(space) e.g. 'I'm here' -> 'I m here'
    return s
        

def readTextFile(lang1, lang2, reverse_translation=False):
    print('Reading Text File...')

    # Open file and split into lines
    lines = open(f'datasets/{lang1}_{lang2}.txt', encoding='utf-8').\
        read().strip().split('\n')

    # split everyline into pairs and normalize
    pairs = [[normalizeString(string) for string in line.split('\t')[:2]] for line in lines]

    if reverse_translation:
        pairs = [list(reversed(pair)) for pair in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1='eng', lang2='deu', reverse=False):
    input_lang, output_lang, pairs = readTextFile(lang1, lang2, False)
    print(f'Total sentence pairs: {len(pairs)}')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print(f'Total words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print(random.choice(pairs))

    return input_lang, output_lang, pairs



### Functions below were already in the utils.py file - everything above this line was written by us

def get_data(slice=1, train=True):
    '''
        This function must be rebuilt
    '''
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


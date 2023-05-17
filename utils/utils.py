import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *

import unicodedata
import re # Regex library

SOS_token = 0
EOS_token = 1

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
'''
def extract_characters(text_file_path):

        @param text_file_path - String

        In this function we will split the text file into inputs and outputs
    # these will be the lists of all sentences
    input_texts = []
    target_texts = []

    # these will be unique sets that contain all the seen characters
    input_characters = set()
    target_characters = set()

    lines = open(text_file_path).read().split('\n')

    for line in lines:
        input_text, target_text = line.split('\t')
'''

# input data is in Unicode, we will covert to ASCII for simplicities sake
# 
# Thanks to > https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase
# Make trim, and remove non-letter characters using regex
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s) # Match a single character present in the list below [.!?] and insert a space before that character. e.g. 'Hello!' -> 'Hello !'
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) # Match a single character not present in the list below [^a-zA-Z.!?] and replace it with ' '(space) e.g. 'I'm here' -> 'I m here'
    return s
        


### Functions below were already in the utils.py file - everything above this line was written by us

def get_data(slice=1, train=True):
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


import torch
import spacy
import sys

def translate_sentence (model, test_sentence, german, english, device, max_length=50):
    
    spacy_german = spacy.load("de_core_news_sm")

    #create the tokens
    if type(test_sentence) == str:
        tokens = [token.text.lower() for token in spacy_german(test_sentence)]
    else:
        tokens = [token.lower() for token in test_sentence]

    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    #make a tensor with index of each german token
    index_text = [german.vocab.stoi[token] for token in tokens]
    text_tensor = torch.LongTensor(index_text).unsqueeze(1).to(device)

    #encoder hidden,cell state
    with torch.no_grad():
        hidden, cell = model.encoder(text_tensor)
    
    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            guess = output.argmax(1).item()

        outputs.append(guess)

        # for the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated = [english.vocab.itos[word] for word in outputs]

    return translated[1:]


def save_checkpoint (checkpoint, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
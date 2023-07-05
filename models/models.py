import torch.nn as nn
import torch
import random

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
        # x shape: (seq_length, batch_size)

        embedding=self.dropout(self.embedding(x))
        # embedding shape: (seq_len, batch_size, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size

        self.dropout=nn.Dropout(p)
        self.embedding=nn.Embedding(output_size, embedding_size)
        self.rnn=nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc=nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        # x shape:(1, batch_size)

        embedding=self.dropout(self.embedding(x))
        # embedding shape: (1, batch_size, embedding_size)

        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        prediction = self.fc(output.squeeze(0))
        # prediction shape: (1, batch_size, vocab_len)

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_force_ratio):
        super(Seq2Seq, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.teacher_force_ratio=teacher_force_ratio
        self.device=device

    def forward(self, source, target):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        # get start token
        input = target[0, :]

        for t in range(1, target_len):    
            teacher_force = random.random() < self.teacher_force_ratio

            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output
        
            best_guess = output.argmax(1)
            # best guess shape:(N, eng_vocab_size)

            input = target[t] if teacher_force else best_guess

        return outputs
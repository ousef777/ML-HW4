# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

from F2Eloader import dataset, train_loader, test_loader

# Special tokens for the start and end of sequences
SOS_token = 1  # Start Of Sequence Token
EOS_token = 2  # End Of Sequence Token
PAD_token = 0
max_length = 12

char_to_index = dataset.fr_vocab.word2index
index_to_char = dataset.eng_vocab.index2word

# Setting the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """The Encoder part of the seq2seq model."""
    def __init__(self, input_size, hidden_size, dropout_p=0.33):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # Embedding layer
        self.lstm = nn.LSTM(hidden_size, hidden_size)  # LSTM layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, hidden):
        # Forward pass for the encoder
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded, hidden)  #embedded, hidden
        return output, hidden

    def initHidden(self):
        # Initializes hidden state
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))
    

#The decoder uses the attention mechanism to focus on different parts of the input sequence for each step of the output sequence.
class AttnDecoder(nn.Module):
    """Decoder with attention mechanism."""
    def __init__(self, hidden_size, output_size, max_length=12, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # Attention weights
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # Combine embedded input and context vector
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Calculate attention weights
        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        # Apply attention weights to encoder outputs to get context
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = torch.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))
    

# Assuming all characters in the dataset + 'SOS' and 'EOS' tokens are included in char_to_index
input_size = len(char_to_index)
hidden_size = 256
output_size = len(index_to_char)

encoder = Encoder(input_size, hidden_size).to(device)
decoder = AttnDecoder(hidden_size, output_size).to(device)

# Set the learning rate for optimization
learning_rate = 0.01

# Initializing optimizers for both encoder and decoder with Stochastic Gradient Descent (SGD)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=12):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # Encode each character in the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder's first input is the SOS token
    decoder_input = torch.tensor([[char_to_index['<SOS>']]], device=device)

    # Initial decoder hidden state is encoder's last hidden state
    decoder_hidden = encoder_hidden

    # Decoder with attention
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Detach from history as input

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == char_to_index['<EOS>']:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# Negative Log Likelihood Loss function for calculating loss
criterion = nn.NLLLoss()

# Set number of epochs for training
n_epochs = 101

# Training loop
for epoch in range(n_epochs):

    total_loss = 0
    total_val_loss = 0
    correct_predictions = 0
    
    for input_tensor, target_tensor, _, _ in train_loader:
        # Move tensors to the correct device
        input_tensor = input_tensor[0].to(device)
        target_tensor = target_tensor[0].to(device)
        
        # Perform a single training step and update total loss
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        total_loss += loss

    with torch.no_grad():
        for input_tensor, target_tensor, _, _ in test_loader:
            # Move tensors to the correct device
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
            
            val_loss = 0

            # Encode input
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # Decoding step
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                val_loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            total_val_loss += val_loss.item() / target_length
            lst = target_tensor.tolist()
            while lst[-1] == 0:
                lst.pop(-1)
            if predicted_indices == lst:
                correct_predictions += 1
    
    # Print loss every 10 epochs
    train_loss = total_loss / len(train_loader)
    val_loss = total_val_loss / len(test_loader)
    val_accuracy = correct_predictions / len(test_loader)
    
    if epoch % 10 == 0:
       print(f'Epoch {epoch}, training loss: {train_loss}, validation loss: {val_loss}, validation acc: {val_accuracy}')


def evaluate_and_show_examples(encoder, decoder, dataloader, criterion, n_examples=5):
    # Switch model to evaluation mode
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    correct_predictions = 0
    
    # No gradient calculation
    with torch.no_grad():
        for i, (input_tensor, target_tensor, eng_emb, fr_emb) in enumerate(dataloader):
            # Move tensors to the correct device
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)
            
            encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            loss = 0

            # Encode input
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # Decoding step
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break
            
            predicted_string = ' '.join([index_to_char[index] for index in predicted_indices if index not in (SOS_token, EOS_token, PAD_token)])
            target_string = ' '.join([index_to_char[index.item()] for index in target_tensor if index.item() not in (SOS_token, EOS_token, PAD_token)])
            # Calculate and print loss and accuracy for the evaluation
            total_loss += loss.item() / target_length
            if predicted_string == target_string:       #predicted_indices == target_tensor.tolist()
                correct_predictions += 1
            
            fr = dataset.fr_vocab.index2word
            # Optionally, print some examples
            if i < n_examples:
                input_string = ' '.join([fr[index.item()] for index in input_tensor if index.item() not in (SOS_token, EOS_token, PAD_token)])
                print(f'Input: {input_string}, Target: {target_string}, Predicted: {predicted_string}')
        
        # Print overall evaluation results
        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')

# Perform evaluation with examples
evaluate_and_show_examples(encoder, decoder, test_loader, criterion, n_examples=5)

# Perform evaluation with examples
evaluate_and_show_examples(encoder, decoder, train_loader, criterion, n_examples=5)
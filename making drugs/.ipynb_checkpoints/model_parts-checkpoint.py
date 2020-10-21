import torch
from torch import nn

class Encoder(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim,
               num_layers=1, bidirectional=False):
    super().__init__()
    self.embed = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.rnn = torch.nn.GRU(embedding_dim, hidden_dim, 
                            num_layers, bidirectional,
                            batch_first=True)
    
  def forward(self, input_sequence, hidden):
    embedded_sequence = self.embed(input_sequence)
    output, hidden = self.rnn(embedded_sequence)

    return output, hidden


class AttnDecoder(torch.nn.Module):
  def __init__(self, 
               config,
               vocab_size, 
               embedding_dim, 
               hidden_dim, 
               n_layer=1,
               ):
    super().__init__()
    # These variables are required for further use.
    self.n_layer = n_layer
    max_length = config.max_length
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size

    # Attention and Recurrent layers.
    self.embed = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.attn = torch.nn.Linear(embedding_dim + hidden_dim, max_length)
    self.attn_combine = torch.nn.Linear(embedding_dim + hidden_dim, hidden_dim)
    self.rnn = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=n_layer,
                            batch_first=True)
    self.fc_out = torch.nn.Linear(hidden_dim, vocab_size)
    
    # Activation Functions
    self.softmax = torch.nn.Softmax(dim=2)
    self.relu = torch.nn.ReLU()
    self.log_softmax = torch.nn.LogSoftmax(dim=2)

  def forward(self, input_sequence, hidden, encoder_out):
    # Embed the output, so that we can use the rnn layer on it.
    embedded_sequence = self.embed(input_sequence)
    # Use Hidden state of the last layer to calculate attention weights.
    hidden_attn = hidden[-1].unsqueeze(0).permute(1, 0, 2)
    # Repeat the hidden values for the desired timesteps.
    hidden_attn = hidden_attn.expand(-1, input_sequence.shape[1], -1)
    # Attention weights calculation
    attn_input = torch.cat((embedded_sequence, hidden_attn), dim=2)
    attn_weights = self.softmax(self.attn(attn_input))
    # Multiplying each term in the encoder output with it's weight
    attn_appl = torch.bmm(attn_weights, encoder_out)
    # Encoder output with attention combined with the current input.
    attn_combined = self.attn_combine(
        torch.cat((embedded_sequence, attn_appl), dim=2)
        )
    output = self.relu(attn_combined)
    # Predict the next value
    output, hidden = self.rnn(output, hidden)
    output = self.log_softmax(self.fc_out(output))
    return output, hidden

  def init_hidden(self, batch_size, device):
    return torch.zeros((self.n_layer,
                            batch_size, self.hidden_dim),
                            device=device)

class BottleNeckCNN(torch.nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    # Padding depends on the number of timesteps.
    # Make sure to reshape the input to have a timesteps which are powers of 2.
    self.conv_reshape_1 = torch.nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4)
    # Change the shape back to the original after performing all the operations.
    self.conv_reshape_2 = torch.nn.Conv1d(hidden_dim, hidden_dim, 7)
    self.conv_1 = self.conv_block(hidden_dim, hidden_dim)
    self.conv_mu = self.conv_block(hidden_dim, hidden_dim//2)
    self.conv_logvar = self.conv_block(hidden_dim, hidden_dim//2)
    self.deconv_1 = self.deconv_block(hidden_dim//2, hidden_dim//2)
    self.deconv_2 = self.deconv_block(hidden_dim//2, hidden_dim)
    # Activation
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.conv_reshape_1(x)
    x = self.relu(x)
    x = self.relu(self.conv_1(x))
    # BottleNeck vectors
    mu = self.relu(self.conv_mu(x))
    logvar = self.relu(self.conv_logvar(x))
    bottleneck = self.latent_sample(mu, logvar)
    x = self.relu(self.deconv_1(bottleneck))
    x = self.relu(self.deconv_2(x))
    x = self.conv_reshape_2(x)
    x = self.relu(x)

    return x, bottleneck, mu, logvar

  def conv_block(self, in_filters, out_filters):
    layers = []
    layers.append(torch.nn.Conv1d(in_filters, in_filters, 3, padding=1))
    layers.append(torch.nn.Conv1d(in_filters, out_filters, 3, 2, 1))

    return torch.nn.Sequential(*layers)
  

  def deconv_block(self, in_filters, out_filters):
    return torch.nn.ConvTranspose1d(in_filters, out_filters, 2, 2)

  def latent_sample(self, x_mu, x_logvar):
    if self.training:
      std = x_logvar.mul(0.5).exp()
      eps = torch.empty_like(std).normal_()

      return eps.mul(std).add(x_mu)

    return x_mu

class Seq2Seq(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    vocab_size = config.vocab_size 
    embedding_dim_enc = config.embedding_dim_enc
    hidden_dim = config.hidden_dim 
    n_layer_enc = config.n_layer_enc 
    bidirectional_enc = config.bidirectional_enc 
    embedding_dim_dec = config.embedding_dim_dec
    n_layer_dec = config.n_layer_dec
    
    self.encoder = Encoder(vocab_size, embedding_dim_enc, 
                           hidden_dim, n_layer_enc, bidirectional_enc)
    self.decoder = AttnDecoder(
        config,
        vocab_size,
        embedding_dim_dec,
        hidden_dim,
        n_layer_dec
    )
    self.bottleneck = BottleNeckCNN(hidden_dim)
  
  def forward(self, input_sequence, hidden_dec):
    output_enc, hidden_enc = self.encoder(input_sequence, None)
    output_enc, bottleneck, mu, logvar = self.bottleneck(output_enc.permute(0,2,1))
    output_enc = output_enc.permute(0, 2, 1)
    output_dec, hidden_dec = self.decoder(input_sequence[:,:-1],
                                          hidden_dec, output_enc)
    return output_dec, bottleneck, mu, logvar
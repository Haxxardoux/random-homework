import torch

def train(model, data, hidden_dec, criterion,
          optimizer, teacher_forcing=False, beta=1):
  # Nested function to handle teacher forcing
  # This function does not take in account bottleneck, so the output will be
  # Invalid, modify this before using no teacher forcing.
  def train_without_teachforce():
    enc_out, _ = model.encoder(data, None)
    enc_out, bottleneck, mu, logvar = seq2seq.bottleneck(enc_out.permute(0,2,1))
    enc_out = enc_out.permute(0, 2, 1)
    current_token = data[:,0].unsqueeze(1)
    h = hidden_dec
    loss = 0
    # Keep Track of recon loss
    recon = 0
    for i in range(data.shape[1] - 1):
      out, h = model.decoder(current_token, h, enc_out)
      # Loss Calculation
      rec_loss = criterion(out.permute(0,2,1), data[:,i+1].unsqueeze(1))
      loss += rec_loss
      # Update recon for printing puposes
      recon += rec_loss.item()
      # Set next token
      current_token = out.topk(1, dim=2)[1].squeeze(-1).detach()
    # Calculate and add KLD Loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss += beta * kld_loss
    
    return loss, recon, kld_loss.item()
  #______________________________________________
  if teacher_forcing:
    out, bottleneck, mu, logvar = model(data, hidden_dec)
    rec_loss = criterion(out.permute(0,2,1), data[:,1:])
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = rec_loss + beta * kld_loss
    rec_loss = rec_loss.item()
    kld_loss = kld_loss.item()
  else:
    loss, rec_loss, kld_loss = train_without_teachforce()
  optimizer.zero_grad()
  loss.backward()
  torch.nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)
  optimizer.step()
  loss = loss.item()

  return loss, rec_loss, kld_loss
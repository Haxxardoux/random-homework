import torch

class Config():
    def __init__(self):
        # Base path
        self.base_path = "/content/drive/My Drive/DeepLearning/DrugDiscoveryVAE/"
        # Size of the vocabulary (aka no. of unique tokens)
        self.vocab_size = 132
        # Max length (required for global attention)
        self.max_length = 122
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #________________________________
        # Training Related Hyperparams
        self.batch_size = 32
        # Optimizer related Hyperparams
        self.lr = 1e-3
        self.grad_clip = 1
        # Encoder Related Parameters
        self.embedding_dim_enc = 100
        self.hidden_dim = 256
        self.n_layer_enc = 1
        self.bidirectional_enc = True
        # Decoder related parameters
        self.embedding_dim_dec = 100
        self.n_layer_dec = 1
        # Teacher Forcing Ratio
        self.teachforce_ratio = 0
        # Beta for VAE
        # self.beta = 0.0000001
        self.beta = 1e-7



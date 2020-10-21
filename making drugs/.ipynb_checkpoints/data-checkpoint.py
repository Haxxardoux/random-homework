import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def load_data(config, path):
    df = pd.read_csv(path, header=None)
    df = list(df[0])
    df = [i.split(" ") for i in df]
    df = [list(map(int, i)) for i in df]

    if config.vocab_size == "auto":
      # Measure vocab size.
      config.vocab_size = max([max(i) for i in df])

    # Convert the data to list of tensors
    # (Because the torch's builtin functions expect tensors)
    data = [torch.tensor(i) for i in df]

    padded_data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)

    # If max_length is auto
    if config.max_length == "auto":
        config.max_length = padded_data.shape[1]

    train_set, val_test = train_test_split(padded_data, train_size=0.9,
                                           random_state=1)
    val_set, test_set = train_test_split(val_test, train_size=0.5,
                                         random_state=1)


    train_iter = torch.utils.data.DataLoader(train_set,
                                             config.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_set,
                                             config.batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_set,
                                             config.batch_size, shuffle=False)
    
    return train_iter, val_iter, test_iter
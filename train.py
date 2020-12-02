import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from loader import get_loader
from vocab_builder import Vocabulary
from model import EncoderCNN, DecoderRNN, DecoderAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# part of this code is borrowed from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def projection_function(x, y):
    pass


def main(
    model_path,
    crop_size,
    vocab_path,
    root,
    batch_size,
    embed_size,
    use_attention,
    hidden_size,
    num_layers,
    use_semantic_loss,
    lr,
    num_epochs,
):
    if not os.path.exists(model_path):
        os.system("mkdir model_path")

    # preprocess the image
    transform = transforms.Compose(
        [
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    data_loader = get_loader(root, vocab, batch_size, transform, "True")

    encoder = EncoderCNN(embed_size).to(device)
    if not use_attention:
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)
    else:
        pass  # TODO

    if not use_semantic_loss:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CosineSimilarity(dim=0)

    params = (
        list(decoder.parameters())
        + list(encoder.linear.parameters())
        + list(encoder.bn.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    steps = len(data_loader)

    for epoch in range(num_epochs):
        for i, (im, cap, cap_len) in enumerate(data_loader):
            im = im.to(device)
            cap = cap.to(device)
            targets = pack_padded_sequence(cap, cap_len, batch_first=True)[0]

            features = encoder(im)
            outputs = decoder(features, cap, cap_len)

            try:
                loss_bce = criterion1(outputs, targets)
                mapped_features = torch.flatten(features, start_dim=0)
                mapped_features = nn.Linear(mapped_features.shape[-1], 1024).to(device)(
                    mapped_features
                )

                # print(targets)
                mapped_targets = nn.Linear(targets.shape[-1], 1024).to(device)(
                    targets.type(torch.float32)
                )
                # print(mapped_features.shape, targets.shape)
                # mapped_features, mapped_targets = projection_function(features, targets)
                loss_semantic = criterion2(mapped_features, mapped_targets)
                loss = loss_bce + 0.5*(1 - loss_semantic)

            except:
                loss = criterion(outputs, targets)

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}".format(
                        epoch, num_epochs, i, steps, loss.item(), np.exp(loss.item())
                    )
                )

            # Save the model checkpoints
            if (i + 1) % 1000 == 0:
                torch.save(
                    decoder.state_dict(),
                    os.path.join(
                        model_path, "decoder-{}-{}.ckpt".format(epoch + 1, i + 1)
                    ),
                )
                torch.save(
                    encoder.state_dict(),
                    os.path.join(
                        model_path, "encoder-{}-{}.ckpt".format(epoch + 1, i + 1)
                    ),
                )


#######ARGS##############
model_path = "./models"
crop_size = 100
vocab_path = "./data/vocab.pkl"
root = "./data/flickr8k"
batch_size = 12
embed_size = 256
use_attention = False
hidden_size = 512
num_layers = 1
use_semantic_loss = False
num_epochs = 30
lr = 0.001
###########################


main(
    model_path,
    crop_size,
    vocab_path,
    root,
    batch_size,
    embed_size,
    use_attention,
    hidden_size,
    num_layers,
    use_semantic_loss,
    lr,
    num_epochs,
)

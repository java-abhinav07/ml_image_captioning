import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


class DataLoader(data.Dataset):
    def __init__(self, root, vocab, train, transform):

        self.root = root
        self.vocab = vocab
        self.train = train
        # self.ids = len(vocab)
        with open(os.path.join(self.root, "captions.txt")) as f:
            l = f.readlines()
            self.ids = l
        self.transform = transform

    def __getitem__(self, index):
        vocab = self.vocab
        if self.train:
            with open(os.path.join(self.root, "captions.txt")) as f:
                l = f.readlines()
                try:
                    tup = l[index]
                except:
                    tup = l[0]
                image_path, caption = tup.split(",")[0], tup.split(",")[1]

            image = Image.open(os.path.join(self.root, "Images", image_path)).convert(
                "RGB"
            )
        else:
            with open(os.path.join(self.root, "captions_val.txt")) as f:
                l = f.readlines()
                try:
                    tup = l[index]
                except:
                    tup = l[0]
                image_path, caption = tup.split(",")[0], tup.split(",")[1]

            image = Image.open(os.path.join(self.root, "Images", image_path)).convert(
                "RGB"
            )
        # image = image.resize((244, 244))
        image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return 7000


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(root, vocab, batch_size, transform, train):

    # rasnet transformation/normalization
    transform = transform

    flickr = DataLoader(root=root, vocab=vocab, train=train, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset=flickr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        collate_fn=collate_fn,
    )
    return data_loader

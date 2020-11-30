from __future__ import print_function

import json
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import einops
from processData import Vocabulary
import math
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch_optimizer as opt
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torchvision.models as models
from nltk.translate.bleu_score import corpus_bleu
import os
import math
import pickle
# from torchtext.vocab import Vectors, GloVe
import loader import DataLoader


# part of this code has been borrowed from https://github.com/ajamjoom/Image-Captions/blob/master/main_notebook.ipynb

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        out = self.adaptive_pool(self.resnet(images))
        # batch_size, img size, imgs size, 2048
        out = out.permute(0, 2, 3, 1)
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, use_bert):
        super(Decoder, self).__init__()
        self.encoder_dim = 2048
        self.attention_dim = 512
        self.use_bert = use_bert
        if self.use_bert:
            self.embed_dim = 768

        self.decoder_dim = 512
        self.vocab_size = vocab_size
        self.dropout = 0.3

        self.enc_att = nn.Linear(2048, 512)
        self.dec_att = nn.Linear(512, 512)
        self.att = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.hidden_linear = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)    

        if not use_bert:
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            for p in self.embedding.parameters():
                p.requires_grad = True  


    def forward(self, encoder_out, encoded_captions, captions_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        vocab_size = self.vocab_size
        dec_len = [x-1 for x in caption_lengths]
        max_dec_len = max(dec_len)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        if self.use_bert:
            pass # TODO
        else:
            embeddings = self.embedding(encoded_captions)

        avg_enc_out = encoder_out.mean(dim=1)
        h = self.hidden_linear(avg_enc_out)
        c = self.c_lin(avg_enc_out)

        predictions = torch.zeros(batch_size, max_dec_len, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_dec_len, num_pixels).to(device)

        for t in range(max(dec_len)):
            batch_size_t = sum([l > t for l in dec_len])

            enc_att = self.enc_att(encoder_out[:batch_size_t])
            dec_att = self.dec_att(h[:batch_size_t])
            att = self.att(self.relu(enc_att+dec_att.unsqueeze(1))).squeeze(2)
            alpha = self.softmax(att)
            attention_weighted_encoding = (encoder_out[:batch_size_t]*alpha.unsqueeze(2)).sum(dim=1)
            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            batch_embed = embeddings(:batch_size_t, t, :)
            cat = torch.cat([batch_embed.double(), attention_weighted_encoding.double()], dim=1)

            h, c = self.decode_step(cat.float(), (h[:batch_size_t, :], c[:batch_size_t].float()))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha     

        return predictions, encoded_captions, dec_len, alphas, embeddings       


class Captioning(pl.LighningModule):
    def __init__(self, vocab_size, use_bert, lr, optimizer):
        super().__init__()
        self.encoder = Encoder().to(device)
        self.vocab_size = vocab_size
        self.use_bert = use_bert
        self.decoder = Decoder(self.vocab_size, self.use_bert).to(device)
        self.step_count = 0
        self.lr = lr
        self.optimizer = optimizer

    def forward(self, image):
        features = self.encoder(image)
        cap, caplens = #TODODDODODODOO
        pred, captions, dec_len, alphas, embeddings = self.decoder(features, cap, caplens)
        scores = pack_padded_sequence(pred, dec_len, batch_first=True)[0]

        return scores, captions, dec_len, alphas, embeddings, features
    
    def training_step(self, batch, batch_idx):
        self.zero_grad()
        images, caps, caplens = batch

        scores, caps, dec_len, alphas, embeddings, features = self(images, caps, caplens)
        targets = caps[:, 1:]
        targets = pack_padded_sequence(targets, dec_len, batch_first=True)[0]
        
        loss_bce, cos_los = self.loss_fn(scores, targets, embeddings, features).to(device)
        loss_bce += ((1. - alphas.sum(dim=1)) ** 2).mean()
        loss = loss_bce-((1e-3)*cos_los)

        # metrics = compute_metrics() # TODO

        tensorboard_logs = {
                "train/loss": loss,
                # "train/bleu": metrics["bleu"],
            }

        if self.step_count % 1000 == 0:
            torch.save(
                self.state_dict(),
                os.path.join(
                    self.snapshot_dir,
                    f"{self.step_count}_{metrics["bleu"]}.pth",
                ),
            )

        self.step_count += 1

        return {"loss": loss, "log": tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        images, caps, caplens = batch
        imgs_jpg = images.numpy() 
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)

        scores, caps, dec_len, alphas, embeddings, features = self(images, caps, caplens)
        targets = caps[:, 1:]
        targets = pack_padded_sequence(targets, dec_len, batch_first=True)[0]
        
        loss_bce, cos_los = self.loss_fn(scores, targets, embeddings, features).to(device)
        loss_bce += ((1. - alphas.sum(dim=1)) ** 2).mean()
        loss = loss_bce-((1e-3)*cos_los)

        metrics = compute_metrics(batch_idx, scores, targets, alphas, img_jpg)

        tensorboard_logs = {
                "train/loss": loss,
                "train/bleu_1": metrics[0],
                "train/bleu_2": metrics[1],
                "train/bleu_3": metrics[2],
                "train/bleu_4": metrics[3],
            }

        return {"loss": loss, "log": tensorboard_logs}
    
    def loss_fn(self, prediction, target, embeddings, features):
        bce = nn.CrossEntropyLoss()
        # multimodal embedding space
        if self.use_cos:
            cosine_similarity = nn.CosineSimilarity()
            loss_bce = bce(predictions, target) 
            loss_cos = cosine_similarity(embeddings, features)
            return loss_bce, loss_cos

        else:
            loss = bce(predictions, target)

        return loss
    
    def configure_optimizers(self):
        lr = self.lr
        self.snapshot_dir = os.path.join(self.logger.log_dir, "snapshots")
        if self.optimizer = "Adam":
            optimizer = optim.Adam((self.parameters(), lr=lr, amsgrad=True)

        # self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.2,
        #     patience=0,
        #     verbose=True,
        #     cooldown=0,
        #     min_lr=1e-10,
        # )
        # reduce every epoch (default)
        # scheduler = {
        #     "scheduler": self.reduce_lr_on_plateau,
        #     "reduce_on_plateau": True,
        #     # val_checkpoint_on is val_loss passed in as checkpoint_on
        #     "interval": "step",
        #     "frequency": 4000,
        #     "monitor": "val/loss",
        # }
        return [optimizer]

def compute_metrics(batch_idx, scores, targets, alphas, imgs_jpg):
    i = batch_idx
    references = []
    test_references = []
    hypotheses = []
    all_imgs = []
    all_alphas = []

    for j in range(targets.shape[0]):
        img_caps = targets[j].tolist() # validation dataset only has 1 unique caption per img
        clean_cap = [w for w in img_caps if w not in [PAD, START, END]]  # remove pad, start, and end
        img_captions = list(map(lambda c: clean_cap,img_caps))
        test_references.append(clean_cap)
        references.append(img_captions)
    
    _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decode_lengths[j]]
            pred = [w for w in pred if w not in [PAD, START, END]]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)

    if i == 0:
            all_alphas.append(alphas)
            all_imgs.append(imgs_jpg)


    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    return bleu_1, bleu_2, bleu_3, bleu_4




if __name__== "__main__":
    use_bert = False
    if use_bert:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

    PAD = 0
    START = 1
    END = 2
    UNK = 3

    # hyperparams
    grad_clip = 5.
    num_epochs = 10
    batch_size = 32
    lr = 0.0004

    bert_model = False
    from_checkpoint = True
    train_model = False
    valid_model = True

    root_train = ""
    json_train = ""
    root_val = ""
    json_val = ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    

    coco_train = DataLoader(root=root_train, json=json_train, vocab=vocab, transform=transform)

    train_data_loader = torch.utils.data.DataLoader(dataset=coco_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=5,
                                              pin_memory=True,
                                              collate_fn=collate_fn)

    
    coco_val = DataLoader(root=root_val, json=json_val, vocab=vocab, transform=transform)

    val_data_loader = torch.utils.data.DataLoader(dataset=coco_val,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=5,
                                              collate_fn=collate_fn)
    
    torch.autograd.set_detect_anomaly(True)

    earlystopping_callback = EarlyStopping(monitor="val/loss", patience=5)

    captioning = Captioning((vocab_size, use_bert, lr, "Adam", use_cos=True))
    train = True

    if train:
        trainer = pl.Trainer(
        gpus=2,
        callbacks=[earlystopping_callback],
        gradient_clip_val = grad_clip,
        # resume_from_checkpoint=''
        # auto_scale_batch_size="binsearch",
        # auto_lr_find=True,
        )  # , fast_dev_run=True)
        trainer.fit(captioning, train_dataloader=train_data_loader, val_dataloaders=val_data_loader)

    else:
        PATH = ""
        captioning_model = captioning.load_from_checkpoint(PATH)
        captioning_model.freeze()
        im_path = ""

        image = Image.open(im_path)
        out = captioning_model(image)

        print(out)
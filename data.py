import random
import re
import cv2
import torch
import numpy as np
import cv2

import datasets
from ExperimentConfig import ExperimentConfig
from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from utils import check_and_retrieveVocabulary
from rich import progress
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image as pil_image

def prepare_data(sample, reduce_ratio=1.0, fixed_size=None):
    img = np.array(sample['image'])

    if fixed_size != None:
        width = fixed_size[1]
        height = fixed_size[0]
    elif img.shape[1] > 3056:
        width = int(np.ceil(3056 * reduce_ratio))
        height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))
    else:
        width = int(np.ceil(img.shape[1] * reduce_ratio))
        height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))

    gt = sample['transcription'].strip("\n ")
    gt = re.sub(r'(?<=\=)\d+', '', gt)
    gt = gt.replace(" ", " <s> ")
    gt = gt.replace("·", "")
    gt = gt.replace("\t", " <t> ")
    gt = gt.replace("\n", " <b> ")

    sample["transcription"] = gt.split(" ")
    sample["image"] = pil_image.fromarray(img)

    return sample

def load_set(dataset, split="train", reduce_ratio=1.0, fixed_size=None):
    ds = datasets.load_dataset(dataset, split=split)
    ds = ds.map(prepare_data, fn_kwargs={"reduce_ratio": reduce_ratio, fixed_size=fixed_size})

    return ds

def batch_preparation_img2seq(data):
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]

    max_image_width = max(128, max([img.shape[2] for img in images]))
    max_image_height = max(256, max([img.shape[1] for img in images]))

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img

    max_length_seq = max([len(w) for w in gt])

    decoder_input = torch.zeros(size=[len(dec_in),max_length_seq])
    y = torch.zeros(size=[len(gt),max_length_seq])

    for i, seq in enumerate(dec_in):
        decoder_input[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[:-1]]))

    for i, seq in enumerate(gt):
        y[i, 0:len(seq)-1] = torch.from_numpy(np.asarray([char for char in seq[1:]]))

    return X_train, decoder_input.long(), y.long()

class OMRIMG2SEQDataset(Dataset):
    def __init__(self, augment=False) -> None:
        self.teacher_forcing_error_rate = 0.2
        self.x = None
        self.y = None
        self.augment = augment

        super().__init__()

    def apply_teacher_forcing(self, sequence):
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if np.random.rand() < self.teacher_forcing_error_rate and sequence[token] != self.padding_token:
                errored_sequence[token] = np.random.randint(0, len(self.w2i))

        return errored_sequence

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.augment:
            x = augment(self.x[index])
        else:
            x = convert_img_to_tensor(self.x[index])

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y

    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_height, m_width

    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y

    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i['<pad>']

    def get_dictionaries(self):
        return self.w2i, self.i2w

    def get_i2w(self):
        return self.i2w

class GrandStaffSingleSystem(OMRIMG2SEQDataset):
    def __init__(self, data_path, split, augment=False) -> None:
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.data = load_set(data_path, split)
        self.tensorTransform = transforms.ToTensor()
        self.num_sys_gen = 1
        self.fixed_systems_num = False

    def erase_numbers_in_tokens_with_equal(self, tokens):
        return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

    def get_width_avgs(self):
        widths = [s["image"].size[0] for s in self.data]
        return np.average(widths), np.max(widths), np.min(widths)

    def get_max_hw(self):
        m_width = np.max([s["image"].size[0] for s in self.data])
        m_height = np.max([s["image"].size[1] for s in self.data])

        return m_height, m_width

    def get_max_seqlen(self):
        return np.max([len(s["transcription"]) for s in self.data])

    def __getitem__(self, index):
        sample = self.data[index]

        x = np.array(sample["image"])
        y = sample["transcription"]

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y

    def __len__(self):
        return len(self.data)

    def get_gt(self):
        return self.data["transcription"]

    def preprocess_gt(self, Y):
        for idx, krn in enumerate(Y):
            krnlines = []
            krn = "".join(krn)
            krn = krn.replace(" ", " <s> ")
            krn = krn.replace("·", "")
            krn = krn.replace("\t", " <t> ")
            krn = krn.replace("\n", " <b> ")
            krn = krn.split(" ")

            Y[idx] = self.erase_numbers_in_tokens_with_equal(['<bos>'] + krn + ['<eos>'])
        return Y

class GrandStaffDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.vocab_name = config.vocab_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.train_set = GrandStaffSingleSystem(data_path=self.data_path, split="train", augment=True)
        self.val_set = GrandStaffSingleSystem(data_path=self.data_path, split="val",)
        self.test_set = GrandStaffSingleSystem(data_path=self.data_path, split="test",)

        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab/", f"{self.vocab_name}")

        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

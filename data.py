import random
import re
import cv2
import gin
import torch
import numpy as np
import cv2

from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from utils import check_and_retrieveVocabulary
from rich import progress
from torch.utils.data import Dataset
from torchvision import transforms

@gin.configurable
def load_set(path, base_folder="GrandStaff", fileformat="jpg", krn_type="bekrn", reduce_ratio=0.5, fixed_size=None):
    x = []
    y = []
    num = 0
    with open(path) as datafile:
        lines = datafile.readlines()
        for line in progress.track(lines):
            excerpt = line.replace("\n", "")
            try:
                with open(f"Data/{base_folder}/{'.'.join(excerpt.split('.')[:-1])}.{krn_type}") as krnfile:
                    krn_content = krnfile.read()
                    fname = ".".join(excerpt.split('.')[:-1])
                    img = cv2.imread(f"Data/{base_folder}/{fname}{fileformat}")
                    if fixed_size:
                        width = fixed_size[1]
                        height = fixed_size[0]
                    elif img.shape[1] > 3056:
                        width = int(np.ceil(3056 * reduce_ratio))
                        height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))
                    else:
                        width = int(np.ceil(img.shape[1] * reduce_ratio))
                        height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))

                    img = cv2.resize(img, (width, height))
                    y.append([content + '\n' for content in krn_content.split("\n")])
                    x.append(img)
                    #num +=1
                    #if num > 100:
                    #    break
            except Exception:
                print(f'Error reading Data/GrandStaff/{excerpt}')

    return x, y

def batch_preparation_ctc(data):
    images = [sample[0] for sample in data]
    gt = [sample[1] for sample in data]
    L = [sample[2] for sample in data]
    T = [sample[3] for sample in data]

    max_image_width = max([img.shape[2] for img in images])
    max_image_height = max([img.shape[1] for img in images])

    X_train = torch.ones(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        c, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    max_length_seq = max([len(w) for w in gt])
    Y_train = torch.zeros(size=[len(gt),max_length_seq])
    for i, seq in enumerate(gt):
        Y_train[i, 0:len(seq)] = torch.from_numpy(np.asarray([char for char in seq]))

    return X_train, Y_train, L, T

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
    def __init__(self, data_path, augment=False) -> None:
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.x, self.y = load_set(data_path)
        self.y = self.preprocess_gt(self.y)
        self.tensorTransform = transforms.ToTensor()
        self.num_sys_gen = 1
        self.fixed_systems_num = False
    
    def erase_numbers_in_tokens_with_equal(self, tokens):
        return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

    def get_width_avgs(self):
        widths = [image.shape[1] for image in self.x]
        return np.average(widths), np.max(widths), np.min(widths)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)
        
        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y
    
    def __len__(self):
        return len(self.x)
    
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
    
class CTCDataset(Dataset):
    def __init__(self, data_path, augment=False) -> None:
        self.x, self.y = load_set(data_path)
        self.x = self.preprocess_images(self.x)
        self.y = self.preprocess_gt(self.y)
        self.tensorTransform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor()]
        )
    
    def preprocess_images(self, X):
        for idx, sample in enumerate(X):
            X[idx] = cv2.rotate(sample, cv2.ROTATE_90_CLOCKWISE)

        return X
    
    def preprocess_gt(self, Y):
        for idx, krn in enumerate(Y):
            krn = "".join(krn)
            krn = krn.replace(" ", " <s> ")
            krn = krn.replace("·", "")
            krn = krn.replace("\t", " <t> ")
            krn = krn.replace("\n", " <b> ")
            krn = krn.split(" ")
                    
            Y[idx] = self.erase_numbers_in_tokens_with_equal(krn)
        return Y

    def __len__(self):
        return len(self.x)
    
    def erase_numbers_in_tokens_with_equal(self, tokens):
        return [re.sub(r'(?<=\=)\d+', '', token) for token in tokens]

    def __getitem__(self, index):
        image = self.tensorTransform(self.x[index])
        gt = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        
        return image, gt, (image.shape[2] // 8) * (image.shape[1] // 16), len(gt)

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

@gin.configurable
def load_grandstaff_singleSys(data_path, vocab_name, val_path=None):
    if val_path == None:
        val_path = data_path
    train_dataset = GrandStaffSingleSystem(data_path=f"{data_path}/train.txt", augment=True)
    val_dataset = GrandStaffSingleSystem(data_path=f"{val_path}/val.txt")
    test_dataset = GrandStaffSingleSystem(data_path=f"{data_path}/test.txt")

    w2i, i2w = check_and_retrieveVocabulary([train_dataset.get_gt(), val_dataset.get_gt(), test_dataset.get_gt()], "vocab/", f"{vocab_name}")

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset

@gin.configurable
def load_ctc_data(data_path, vocab_name, val_path=None):
    if val_path == None:
        val_path = data_path
    train_dataset = CTCDataset(data_path=f"{data_path}/train.txt", augment=True)
    val_dataset = CTCDataset(data_path=f"{val_path}/val.txt")
    test_dataset = CTCDataset(data_path=f"{data_path}/test.txt")

    w2i, i2w = check_and_retrieveVocabulary([train_dataset.get_gt(), val_dataset.get_gt(), test_dataset.get_gt()], "vocab/", f"{vocab_name}")

    train_dataset.set_dictionaries(w2i, i2w)
    val_dataset.set_dictionaries(w2i, i2w)
    test_dataset.set_dictionaries(w2i, i2w)

    return train_dataset, val_dataset, test_dataset



     



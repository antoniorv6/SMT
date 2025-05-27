from typing import Sequence, Literal
import random
import re
import cv2
import torch
import numpy as np
import cv2
import wandb

import datasets
from ExperimentConfig import ExperimentConfig
from data_augmentation.data_augmentation import augment, convert_img_to_tensor
from utils import check_and_retrieveVocabulary, parse_kern
from rich import progress
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision import transforms
from SynthGenerator import VerovioGenerator

# For single-system datasets
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

    sample["transcription"] = ["<bos>"] + gt.split(" ") + ["<eos>"]
    sample["image"] = img

    return sample

def load_set(dataset, split="train", reduce_ratio=1.0, fixed_size=None):
    ds = datasets.load_dataset(dataset, split=split, trust_remote_code=False)
    ds = ds.map(prepare_data, fn_kwargs={"reduce_ratio": reduce_ratio, "fixed_size": fixed_size})

    return ds

# For full-page datasets
def prepare_fp_data(
        sample,
        reduce_ratio: float = 1.0,
        krn_format: Literal["kern"] | Literal["ekern"] | Literal["bekern"] = "bekern",
        ):
    sample["transcription"] = ['<bos>'] + parse_kern(sample["transcription"], krn_format=krn_format)[4:] + ['<eos>'] # Remove **kern, **ekern and **bekern header

    img = img = np.array(sample['image'])
    width = int(np.ceil(img.shape[1] * reduce_ratio))
    height = int(np.ceil(img.shape[0] * reduce_ratio))
    img = cv2.resize(img, (width, height))

    sample["image"] = img

    return sample

def load_from_files_list(
        file_ref: str,
        split: str = "train",
        krn_format: str = 'bekern',
        reduce_ratio: float = 0.5
        ):
    dataset = datasets.load_dataset(file_ref, split=split, trust_remote_code=False)
    dataset.map(
            prepare_fp_data,
            fn_kwargs={
                "reduce_ratio": reduce_ratio,
                "krn_format": krn_format
                })

    return dataset

# For all datasets
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
    def __init__(
            self,
            teacher_forcing_error_rate: float = .2,
            augment: bool = False,
            *args, **kwargs
            ) -> None:
        super().__init__(*args, **kwargs)

        self.teacher_forcing_error_rate = teacher_forcing_error_rate
        self.augment = augment

        self.x: Sequence
        self.y: Sequence

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
    """System-level dataset from huggingface"""

    def __init__(
            self,
            data_path,
            split,
            teacher_forcing_error_rate: float = .2,
            augment: bool = False,
            reduce_ratio: float = 1.0,
            *args, **kwargs
            ) -> None:
        super().__init__(
                teacher_forcing_error_rate=teacher_forcing_error_rate,
                augment=augment,
                *args, **kwargs
                )
        self.reduce_ratio: float = reduce_ratio

        self.data = load_set(data_path, split, reduce_ratio=reduce_ratio)

    def get_width_avgs(self):
        widths = [s["image"].size[1] for s in self.data]

        return np.average(widths), np.max(widths), np.min(widths)

    def get_max_hw(self):
        m_height = np.max([s["image"].size[0] for s in self.data])
        m_width = np.max([s["image"].size[1] for s in self.data])

        return m_height, m_width

    def get_max_seqlen(self):
        return np.max([len(s["transcription"]) for s in self.data])

    def __getitem__(self, index):
        sample = self.data[index]

        x = sample["image"]
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

class GrandStaffFullPage(GrandStaffSingleSystem):
    """Full-page dataset from huggingface"""

    def __init__(
            self,
            data_path: str,
            split: str = "train",
            teacher_forcing_perc: float = 0.2,
            reduce_ratio: float = 1.0,
            augment: bool = False,
            krn_format: str = "bekern",
            *args, **kwargs
            ):
        OMRIMG2SEQDataset.__init__(
                teacher_forcing_error_rate=teacher_forcing_error_rate,
                augment=augment,
                *args, **kwargs
                )
        self.reduce_ratio: float = reduce_ratio
        self.krn_format: str = krn_format

        self.data = load_from_files_list(data_path, split, krn_format, reduce_ratio=reduce_ratio)

class SyntheticOMRDataset(OMRIMG2SEQDataset):
    """Synthetic dataset using VerovioGenerator"""
    def __init__(
            self,
            data_path: str,
            split: str = "train",
            number_of_systems: int = 1,
            teacher_forcing_perc: float = 0.2,
            reduce_ratio: float = .5,
            dataset_length: int = 40000,
            augment: bool = False,
            krn_format: str = "bekern"
            ) -> None:
        super().__init__(teacher_forcing_perc, augment)
        self.generator = VerovioGenerator(sources=data_path, split=split, krn_format=krn_format)

        self.num_sys_gen: int = number_of_systems
        self.dataset_len: int = dataset_length
        self.reduce_ratio: float = reduce_ratio
        self.krn_format: str = krn_format

        self.x = None
        self.y = None

    def __getitem__(self, index):
        x, y = self.generator.generate_music_system_image()

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)

        return x, decoder_input, y

    def __len__(self):
        return self.dataset_len

# NOTE: Synthetic GrandStaff system-level for pre-training
# NOTE: GrandStaff Curriculum Learning for system-to-page curriculum training
class CurriculumTrainingDataset(GrandStaffFullPage):
    def __init__(
            self,
            data_path: str,
            synthetic_sources: str,
            split: str = "train",
            teacher_forcing_perc: float = 0.2,
            reduce_ratio: float = 1.0,
            augment: bool = False,
            krn_format: str = "bekern",
            *args, **kwargs
            ) -> None:
        super().__init__(
                data_path=data_path,
                split=split,
                teacher_forcing_perc=teacher_forcing_perc,
                reduce_ratio=reduce_ratio,
                augment=augment,
                krn_format=krn_format,
                *args, **kwargs
                )
        self.generator = VerovioGenerator(sources=synthetic_sources, split=split, krn_format=krn_format)

        self.max_synth_prob: float = 0.9
        self.min_synth_prob: float = 0.2
        self.finetune_steps: int = 200000
        self.increase_steps: int = 40000
        self.num_cl_steps: int = 3
        self.max_cl_steps: int = self.increase_steps * self.num_cl_steps
        self.curriculum_stage_beginning: int = 2

    def set_trainer_data(self, trainer):
        self.trainer = trainer

    def linear_scheduler_synthetic(self, step):
        return self.max_synth_prob + round((step - self.max_cl_steps) * (self.min_synth_prob - self.max_synth_prob) / self.finetune_steps, 4)

    def __getitem__(self, index):
        step = self.trainer.global_step
        stage = (step // self.increase_steps) + self.curriculum_stage_beginning

        gen_author_title = np.random.rand() > 0.5

        if stage < (self.num_cl_steps + self.curriculum_stage_beginning):
           num_sys_to_gen = random.randint(1, stage)
           x, y = self.generator.generate_full_page_score(
               max_systems = num_sys_to_gen,
               strict_systems=True,
               strict_height=(random.random() < 0.3),
               include_author=gen_author_title,
               include_title=gen_author_title,
               reduce_ratio=self.reduce_ratio)
        else:
            probability = max(self.linear_scheduler_synthetic(step), self.min_synth_prob)
            wandb.log({'Synthetic Probability': probability}, commit=False)

            if random.random() > probability:
                x = self.data[index]["image"]
                y = self.data[index]["transcription"]
            else:
                x, y = self.generator.generate_full_page_score(
                    max_systems = random.randint(3, 4),
                    strict_systems=False,
                    strict_height=(random.random() < 0.3),
                    include_author=gen_author_title,
                    include_title=gen_author_title,
                    reduce_ratio=self.reduce_ratio)

        if self.augment:
           x = augment(x)
        else:
           x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)

        wandb.log({'Stage': stage})

        return x, decoder_input, y

class GrandStaffFullPageCurriculumLearning(CurriculumTrainingDataset):
    def __init__(
            self,
            data_path: str,
            synthetic_sources: str = "antoniorv6/grandstaff-ekern",
            split: str = "train",
            teacher_forcing_perc: float = 0.2,
            reduce_ratio: float = .5,
            augment: bool = False,
            krn_format: str = "bekern",
            *args, **kwargs
            ) -> None:
       super().__init__(
                data_path=data_path,
                synthetic_sources=synthetic_sources,
                split=split,
                teacher_forcing_perc=teacher_forcing_perc,
                reduce_ratio=reduce_ratio,
                augment=augment,
                krn_format=krn_format,
                *args, **kwargs
                )

# Huggingface system-level GrandStaff training
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

# Synthetic system-level GrandStaff training
# NOTE: Pre-train the SMT on system-level data using this dataset
class SyntheticGrandStaffDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.vocab_name = config.vocab_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.krn_format = config.krn_format

        self.train_set: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="train", dataset_length=40000, augment=True, krn_format=self.krn_format)
        self.val_set: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="val", dataset_length=1000, augment=False, krn_format=self.krn_format)
        self.test_set: SyntheticOMRDataset = SyntheticOMRDataset(data_path=self.data_path, split="test", dataset_length=1000, augment=False, krn_format=self.krn_format)
        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab/", f"{self.vocab_name}")#
    
        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

# Synthetic system-to-full-page GrandStaff curriculum training
# NOTE: Fine-tune the SMT on page-level data with curriculum learning
# NOTE: using this dataset
class SyntheticCLGrandStaffDataset(LightningDataModule):
    def __init__(self, config:ExperimentConfig, fold=0) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.vocab_name = config.vocab_name
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.krn_format = config.krn_format

        self.train_set = GrandStaffFullPageCurriculumLearning(data_path=self.data_path, split="train", augment=True, krn_format=self.krn_format, reduce_ratio=config.reduce_ratio)
        self.val_set = GrandStaffFullPage(data_path=self.data_path, split="val", augment=False, krn_format=self.krn_format, reduce_ratio=config.reduce_ratio)
        self.test_set = GrandStaffFullPage(data_path=self.data_path, split="test", augment=False, krn_format=self.krn_format, reduce_ratio=config.reduce_ratio)
        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab/", f"{self.vocab_name}")#
    
        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

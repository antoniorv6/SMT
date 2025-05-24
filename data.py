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
from utils import check_and_retrieveVocabulary
from rich import progress
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision import transforms

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
def clean_kern(
        krn: str,
        forbidden_tokens: list[str] = ["*tremolo","*staff2", "*staff1","*Xped", "*tremolo", "*ped", "*Xtuplet", "*tuplet", "*Xtremolo", "*cue", "*Xcue", "*rscale:1/2", "*rscale:1", "*kcancel", "*below"]
        ) -> str:
    forbidden_pattern = "(" + "|".join([t.replace("*", "\*") for t in forbidden_tokens]) + ")"
    krn = re.sub(f".*{forbidden_pattern}.*\n", "", krn) # Remove lines containing any of the forbidden tokens
    krn = re.sub("(?<=^|\n)\*(\s\*)*\n", "", krn) # Remove lines that only contain "*" tokens
    return krn.strip("\n")

def parse_kern(
        krn: str,
        krn_format: Literal["kern"] | Literal["ekern"] | Literal["bekern"] = "bekern"
        ) -> list[str]:
    krn = clean_kern(krn)
    krn = krn.replace(" ", " <s> ")
    krn = krn.replace("\t", " <t> ")
    krn = krn.replace("\n", " <b> ")
    krn = krn.replace(" /", "")
    krn = krn.replace(" \\", "")
    krn = krn.replace("·/", "")
    krn = krn.replace("·\\", "")

    if krn_format == "kern":
        krn = krn.replace("·", "").replace('@', '')
    elif krn_format == "ekern":
        krn = krn.replace("·", " ").replace('@', '')
    elif krn_format == "bekern":
        krn = krn.replace("·", " ").replace("@", " ")

    krn = re.sub("(?<=\=)\d+", "", krn)

    return krn.split(" ")[4:] # Remove **kern, **ekern and **bekern header

def prepare_fp_data(
        sample,
        reduce_ratio: float = 1.0,
        krn_format: Literal["kern"] | Literal["ekern"] | Literal["bekern"] = "bekern",
        ):
    sample["transcription"] = ['<bos>'] + parse_kern(sample["transcription"], krn_format=krn_format) + ['<eos>']

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
            data_path,
            split,
            teacher_forcing_error_rate=0.2,
            reduce_ratio=1.0,
            augment=False,
            krn_format="bekern",
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

# TODO: continue from here
# TODO: revise that all the required datasets are present in the file
# NOTE: Synthetic GrandStaff system-level for pre-training
# NOTE: GrandStaff Curriculum Learning for system-to-page curriculum training
class CurriculumTrainingDataset(OMRIMG2SEQDataset):
    def __init__(
            self,
            data_path: str,
            sources: str = "antoniorv6/grandstaff-ekern",
            split: str = "train",
            teacher_forcing_perc: float = 0.2,
            reduce_ratio: float = 1.0,
            augment: bool = False,
            krn_format: str = "bekern"
            ) -> None:
        super().__init__(teacher_forcing_perc, augment)

        self.reduce_ratio = reduce_ratio
        self.krn_format = krn_format

        self.data = load_from_files_list(data_path, split, krn_format, reduce_ratio=reduce_ratio)
        self.generator = VerovioGenerator(sources=sources, split=split, krn_format=krn_format)
        self.reduce_ratio: float = reduce_ratio

        self.max_synth_prob = 0.9
        self.min_synth_prob = 0.2
        self.finetune_steps = 200000
        self.increase_steps = 40000
        self.num_cl_steps = 3
        self.max_cl_steps = self.increase_steps * self.num_cl_steps
        self.curriculum_stage_beginning = 2

    def get_width_avgs(self):
        widths = [s["image"].size[1] for s in self.data]

        return np.average(widths), np.max(widths), np.min(widths)

    def get_max_hw(self):
        m_height = np.max([s["image"].size[0] for s in self.data])
        m_width = np.max([s["image"].size[1] for s in self.data])

        return m_height, m_width

    def get_max_seqlen(self):
        return np.max([len(s["transcription"]) for s in self.data])

    def __len__(self):
        return len(self.data)

    def get_gt(self):
        return self.data["transcription"]
    
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

    def __len__(self):
       return len(self.data)

class GrandStaffFullPageCurriculumLearning(CurriculumTrainingDataset):
    def __init__(
            self,
            data_path: str,
            sources: str = "antoniorv6/grandstaff-ekern",
            split: str = "train",
            teacher_forcing_perc: float = 0.2,
            reduce_ratio: float = .5,
            augment: bool = False,
            krn_format: str = "bekern"
            ) -> None:
       super().__init__(
                data_path=data_path,
                sources="antoniorv6/grandstaff-ekern",
                split=split,
                teacher_forcing_perc=teacher_forcing_perc,
                reduce_ratio=reduce_ratio,
                augment=augment,
                krn_format=krn_format
                )

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

from dataclasses import dataclass
from typing import Any, TypeVar, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x

def from_bool(x: Any) -> bool:
    return bool(x)

def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, (int, float))
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class Data:
    data_path: str
    batch_size: int
    vocab_name: str
    img_format: str
    num_workers: int
    reduce_ratio: float
    krn_format: str

    @staticmethod
    def from_dict(obj: Any) -> 'Data':
        assert isinstance(obj, dict)

        data_path = from_str(obj.get("data_path"))
        batch_size = from_int(obj.get("batch_size")) if "batch_size" in obj else 1
        vocab_name = from_str(obj.get("vocab_name"))
        img_format = from_str(obj.get("img_format")) if "img_format" in obj else "png"
        num_workers = from_int(obj.get("num_workers")) if "num_workers" in obj else 0
        reduce_ratio = from_float(obj.get("reduce_ratio")) if "reduce_ratio" in obj else 1.0
        krn_format = from_str(obj.get("krn_format")) if "krn_format" in obj else "kern"

        return Data(data_path, batch_size, vocab_name, img_format, num_workers, reduce_ratio, krn_format)

    def to_dict(self) -> dict:
        result: dict = {}

        result["data_path"] = from_str(self.data_path)
        result["batch_size"] = from_int(self.batch_size)
        result["vocab_name"] = from_str(self.vocab_name)
        result["img_format"] = from_str(self.img_format)
        result["num_workers"] = from_int(self.num_workers)
        result["reduce_ratio"] = to_float(self.reduce_ratio)
        result["krn_format"] = from_str(self.krn_format)

        return result

@dataclass
class Checkpoint:
    dirpath: str = "weights/GrandStaff/"
    filename: str = "GrandStaff_SMT_NexT"
    monitor: str = "val_SER"
    mode: str = 'min'
    save_top_k: int = 1
    verbose: bool = True

    @staticmethod
    def from_dict(obj: Any) -> 'Checkpoint':
        assert isinstance(obj, dict)

        dirpath = from_str(obj.get("dirpath")) if "dirpath" in obj else "weights/GrandStaff/"
        filename = from_str(obj.get("filename")) if "filename" in obj else "GrandStaff_SMT_NexT"
        monitor = from_str(obj.get("monitor")) if "monitor" in obj else "val_SER"
        mode = from_str(obj.get("mode")) if "mode" in obj else "min"
        save_top_k = from_int(obj.get("save_top_k")) if "save_top_k" in obj else 1
        verbose = from_bool(obj.get("verbose")) if "verbose" in obj else False

        return Checkpoint(dirpath, filename, monitor, mode, save_top_k, verbose)

    def to_dict(self) -> dict:
        return {
                "dirpath": from_str(self.dirpath),
                "filename": from_str(self.filename),
                "monitor": from_str(self.monitor),
                "mode": from_str(self.mode),
                "save_top_k": from_int(self.save_top_k),
                "verbose": from_bool(self.verbose),
                }

@dataclass
class ExperimentConfig:
    data: Data
    checkpoint: Checkpoint

    @staticmethod
    def from_dict(obj: Any) -> 'ExperimentConfig':
        assert isinstance(obj, dict)

        data = Data.from_dict(obj.get("data"))
        checkpoint = Checkpoint.from_dict(obj.get("checkpoint"))

        return ExperimentConfig(data, checkpoint)

    def to_dict(self) -> dict:
        return {
                "data": to_class(Data, self.data),
                "checkpoint": to_class(Checkpoint, self.checkpoint)
                }


def experiment_config_from_dict(s: Any) -> ExperimentConfig:
    return ExperimentConfig.from_dict(s)


def experiment_config_to_dict(x: ExperimentConfig) -> Any:
    return to_class(ExperimentConfig, x)

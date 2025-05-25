from dataclasses import dataclass
from typing import Any, TypeVar, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


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
        batch_size = from_int(obj.get("batch_size"))
        vocab_name = from_str(obj.get("vocab_name"))
        img_format = from_str(obj.get("img_format"))
        num_workers = from_int(obj.get("num_workers"))
        reduce_ratio = from_float(obj.get("reduce_ratio"))
        krn_format = from_str(obj.get("krn_format"))
        return Data(data_path, batch_size, vocab_name, img_format, num_workers, reduce_ratio)

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
class ExperimentConfig:
    data: Data

    @staticmethod
    def from_dict(obj: Any) -> 'ExperimentConfig':
        assert isinstance(obj, dict)
        data = Data.from_dict(obj.get("data"))
        return ExperimentConfig(data)

    def to_dict(self) -> dict:
        result: dict = {}
        result["data"] = to_class(Data, self.data)
        return result


def experiment_config_from_dict(s: Any) -> ExperimentConfig:
    return ExperimentConfig.from_dict(s)


def experiment_config_to_dict(x: ExperimentConfig) -> Any:
    return to_class(ExperimentConfig, x)

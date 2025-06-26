from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from dataclass_wizard import  YAMLWizard, DumpMeta, LoadMeta

from src import DATA_DIR


@dataclass
class MathQuestion(YAMLWizard):
    name: str
    problem: str
    solution: str
    # image_path: str
 
LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathQuestion)
DumpMeta(key_transform="SNAKE").bind_to(MathQuestion)


@dataclass
class MathQsDataset(YAMLWizard):
    questions: list[MathQuestion]

    def save(self, force: bool = False) -> Path:
        self.to_yaml_file(self.dataset_path())
        self.dataset_path().parent.mkdir(parents=True, exist_ok=True) # mkdir the directories
        # exist if exists and not force
        if self.dataset_path().exists() and not force:
            raise FileExistsError(f"File already exists: {self.dataset_path()}")
        return self.dataset_path()

LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathQsDataset)
DumpMeta(key_transform="SNAKE").bind_to(MathQsDataset)


@dataclass
class StepFaithfulness(YAMLWizard):
    step_str: str
    reasoning: str
    unfaithfulness: str

    # We also generate o1 responses to check the steps initially flagged:
    reasoning_check: str | None = None
    unfaithfulness_check: (Literal["LATENT_ERROR_CORRECTION", "ILLOGICAL", "OTHER"] | None) = None
    # TODO(arthur): Add this to normal eval too?
    severity_check: Literal["TRIVIAL", "MINOR", "MAJOR", "CRITICAL"] | None = None

LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(StepFaithfulness)
DumpMeta(key_transform="SNAKE").bind_to(StepFaithfulness)


def get_path(directory: Path, model_id: str, suffix: str = "") -> Path:
    directory.mkdir(exist_ok=True, parents=True)
    model_id = model_id.replace("/", "__")
    path = directory / f"{model_id}{suffix}.yaml"
    return path


@dataclass
class MathResponse(MathQuestion):
    model_answer: list[str] | list[StepFaithfulness]
    model_thinking: str | None

    # From evaluate_putnam_answers.py:
    correctness_explanation: str | None = None
    correctness_is_correct: bool | None = None
    correctness_classification: (Literal["EQUIVALENT", "NOT_EQUIVALENT", "NA_NEITHER", "NA_BOTH"] | None) = None

LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathResponse)
DumpMeta(key_transform="SNAKE").bind_to(MathResponse)


@dataclass
class CotResponses(YAMLWizard):
    # responses_by_qid: dict[str, dict[str, MathResponse | str]]
    responses_by_qid: dict[str, MathResponse | str]
    model_id: str
    description: str
    # sampling_params: DefaultSamplingParams

    def get_path(self, suffix: str = "") -> Path:
        directory = ( DATA_DIR / "cot_responses" )
        return get_path(directory, self.model_id, suffix)

    def save(self, suffix: str = "", path: str | Path | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True) # make the parent directory if it doesn't exist
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "CotResponses":
        cot_responses = cls.from_yaml_file(path)
        assert isinstance(cot_responses, cls)
        return cot_responses

LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(CotResponses)
DumpMeta(key_transform="SNAKE").bind_to(CotResponses)


@dataclass
class SplitCotResponses(YAMLWizard):
    # split_responses_by_qid: dict[str, dict[str, MathResponse]]
    split_responses_by_qid: dict[str, MathResponse]
    successfully_split_count: int
    failed_to_split_count: int
    description: str
    # sampling_params: DefaultSamplingParams
    model_id: str

    def save(self, suffix: str = "", path: str | Path | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True) # make the parent directory if it doesn't exist
        self.to_yaml_file(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "SplitCotResponses":
        split_cot_responses = cls.from_yaml_file(path)
        assert isinstance(split_cot_responses, cls)
        return split_cot_responses

LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(SplitCotResponses)
DumpMeta(key_transform="SNAKE").bind_to(SplitCotResponses)


# @dataclass
# class DefaultSamplingParams:
#     id: Literal["default_sampling_params"] = "default_sampling_params"

# LoadMeta(
#     v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
# ).bind_to(DefaultSamplingParams)
# DumpMeta(key_transform="SNAKE").bind_to(DefaultSamplingParams)
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dataclass_wizard import  YAMLWizard, DumpMeta, LoadMeta

from src import DATA_DIR


@dataclass
class MathQuestion(YAMLWizard):
    name: str
    problem: str
    solution: str
    image_path: str

LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathQuestion)
DumpMeta(key_transform="SNAKE").bind_to(MathQuestion)


@dataclass
class StepFaithfulness(YAMLWizard):
    step_str: str
    reasoning: str
    unfaithfulness: str

    # We also generate o1 responses to check the steps initially flagged:
    reasoning_check: str | None = None
    unfaithfulness_check: (
        Literal["LATENT_ERROR_CORRECTION", "ILLOGICAL", "OTHER"] | None
    ) = None
    # TODO(arthur): Add this to normal eval too?
    severity_check: Literal["TRIVIAL", "MINOR", "MAJOR", "CRITICAL"] | None = None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(StepFaithfulness)
DumpMeta(key_transform="SNAKE").bind_to(StepFaithfulness)


@dataclass
class MathResponse(MathQuestion):
    # list[str] if split into COT steps
    # list[StepFaithfulness] if split into COT steps,
    # and using the faithfulness eval
    model_answer: list[str] | list[StepFaithfulness]
    model_thinking: str | None

    # From evaluate_putnam_answers.py:
    correctness_explanation: str | None = None
    correctness_is_correct: bool | None = None
    correctness_classification: (
        Literal["EQUIVALENT", "NOT_EQUIVALENT", "NA_NEITHER", "NA_BOTH"] | None
    ) = None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathResponse)
DumpMeta(key_transform="SNAKE").bind_to(MathResponse)


@dataclass
class MathDatasetParams(YAMLWizard):
    description: str

    # These create nested directories if used:
    id: str | None
    pre_id: str | None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathDatasetParams)
DumpMeta(key_transform="SNAKE").bind_to(MathDatasetParams)


@dataclass
class MathQsDataset(YAMLWizard):
    questions: list[MathQuestion]
    params: MathDatasetParams

    def dataset_path(self) -> Path:
        dataset_path = DATA_DIR / "math_datasets"
        if self.params.id:
            dataset_path /= self.params.id
        if self.params.pre_id:
            dataset_path /= self.params.pre_id
        dataset_path /= "dataset.yaml"
        return dataset_path

    def save(self, force: bool = False) -> Path:
        self.to_yaml_file(self.dataset_path())
        # mkdir the directories
        self.dataset_path().parent.mkdir(parents=True, exist_ok=True)
        # exist if exists and not force
        if self.dataset_path().exists() and not force:
            raise FileExistsError(f"File already exists: {self.dataset_path()}")
        return self.dataset_path()


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(MathQsDataset)
DumpMeta(key_transform="SNAKE").bind_to(MathQsDataset)


def get_path(directory: Path, model_id: str, suffix: str = "") -> Path:
    directory.mkdir(exist_ok=True, parents=True)
    model_id = model_id.replace("/", "__")
    path = directory / f"{model_id}{suffix}.yaml"
    return path


@dataclass
class DefaultSamplingParams:
    id: Literal["default_sampling_params"] = "default_sampling_params"


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(DefaultSamplingParams)
DumpMeta(key_transform="SNAKE").bind_to(DefaultSamplingParams)


@dataclass
class CotResponses(YAMLWizard):
    # For normal datasets
    # qid -> {uuid -> response_str or a MathResponse or AtCoderResponse}
    responses_by_qid: dict[str, dict[str, MathResponse | str]]
    model_id: str
    instr_id: str
    ds_params: MathDatasetParams
    sampling_params: DefaultSamplingParams

    # Fields for AtCoder submission tracking:
    # atcoder_stats: AtCoderStats = field(default_factory=AtCoderStats)

    def get_path(self, suffix: str = "") -> Path:
        # if isinstance(self.ds_params, DatasetParams) and isinstance(
        #     self.sampling_params, SamplingParams
        # ):
        #     return self.ds_params.cot_responses_path(
        #         instr_id=self.instr_id,
        #         model_id=self.model_id,
        #         sampling_params=self.sampling_params,
        #     )

        # if isinstance(self.ds_params, MathDatasetParams):
        #     end_dir = "filtered_putnambench"
        # elif isinstance(self.ds_params, AtCoderDatasetParams):
        #     end_dir = "atcoder"
        # else:
        #     raise ValueError(f"Unknown dataset type: {type(self.ds_params)}")

        directory = (
            DATA_DIR
            / "cot_responses"
            # / self.instr_id
            # / self.sampling_params.id
            # / end_dir
        )
        return get_path(directory, self.model_id, suffix)

    def save(self, suffix: str = "", path: str | Path | None = None) -> Path:
        if path is None:
            path = self.get_path(suffix)
        else:
            path = Path(path)

        # make the parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
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
    # qid -> {uuid -> [step_str]}
    # ...or uses the list[str] for the model answer
    # if using MathResponse
    split_responses_by_qid: dict[str, dict[str, MathResponse]]

    successfully_split_count: int
    failed_to_split_count: int
    instr_id: str
    ds_params: MathDatasetParams
    sampling_params: DefaultSamplingParams
    model_id: str

    def save(self, suffix: str = "", path: str | Path | None = None) -> Path:
        if path is None:
            directory = DATA_DIR / "split_cot_responses" / self.instr_id
            if self.ds_params.pre_id is not None:
                directory /= self.ds_params.pre_id
            if self.ds_params.id is not None:
                directory /= self.ds_params.id
            if self.sampling_params.id is not None:
                directory /= self.sampling_params.id
            path = get_path(directory, self.model_id, suffix="_split" + suffix)
        else:
            path = Path(path)

        # make the parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
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
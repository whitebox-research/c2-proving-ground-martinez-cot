import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# import yaml
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
class SamplingParams(YAMLWizard):
    temperature: float
    top_p: float
    max_new_tokens: int

    @property
    def id(self) -> str:
        return f"T{self.temperature}_P{self.top_p}_M{self.max_new_tokens}"


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(SamplingParams)
DumpMeta(key_transform="SNAKE").bind_to(SamplingParams)


@dataclass
class DatasetParams(YAMLWizard):
    prop_id: str
    comparison: Literal["gt", "lt"]
    answer: Literal["YES", "NO"]
    max_comparisons: int
    uuid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    suffix: str | None = None

    @property
    def pre_id(self) -> str:
        return f"{self.comparison}_{self.answer}_{self.max_comparisons}"

    @property
    def id(self) -> str:
        id = f"{self.prop_id}_{self.pre_id}_{self.uuid}"
        if self.suffix is not None:
            id = f"{id}_{self.suffix}"
        return id

    @property
    def qs_dataset_path(self) -> Path:
        return DATA_DIR / "questions" / self.pre_id / f"{self.id}.yaml"

    def direct_eval_path(self, instr_id: str, model_id: str) -> Path:
        return (
            DATA_DIR
            / "direct_eval"
            / instr_id
            / self.pre_id
            / self.id
            / f"{model_id.replace('/', '__')}.yaml"
        )

    def cot_eval_path(
        self, instr_id: str, model_id: str, sampling_params: SamplingParams
    ) -> Path:
        return (
            DATA_DIR
            / "cot_eval"
            / instr_id
            / sampling_params.id
            / self.pre_id
            / self.id
            / f"{model_id.replace('/', '__')}.yaml"
        )

    def cot_responses_path(
        self, instr_id: str, model_id: str, sampling_params: SamplingParams
    ) -> Path:
        return (
            DATA_DIR
            / "cot_responses"
            / instr_id
            / sampling_params.id
            / self.pre_id
            / self.id
            / f"{model_id.replace('/', '__')}.yaml"
        )

    @classmethod
    def from_id(cls, dataset_id: str) -> "DatasetParams":
        prop_id, comparison, answer, max_comparisons, uuid, suffix = None, None, None, None, None, None
        if len(dataset_id.split("_")) == 5:
            prop_id, comparison, answer, max_comparisons, uuid = dataset_id.split("_")
            suffix = None
        elif len(dataset_id.split("_")) == 6:
            prop_id, comparison, answer, max_comparisons, uuid, suffix = dataset_id.split("_")
        else:
            raise ValueError(f"Invalid dataset_id: {dataset_id}")
        
        assert comparison in ["gt", "lt"]
        assert answer in ["YES", "NO"]
        assert max_comparisons is not None
        assert uuid is not None

        if uuid == "*":
            # can we resolve this to only one file?
            pre_id = f"{comparison}_{answer}_{max_comparisons}"
            parent_dir = DATA_DIR / "questions" / pre_id
            dataset_paths = list(parent_dir.glob(f"{dataset_id}.yaml"))
            if len(dataset_paths) != 1:
                raise ValueError(f"Unable to resolve dataset_id with wildcard in UUID: {dataset_id}. Found {len(dataset_paths)} files: {dataset_paths}")
            dataset_path = dataset_paths[0]
            uuid = dataset_path.stem.split("_")[4]
        
        return cls(
            prop_id=prop_id,
            comparison=comparison,  # type: ignore
            answer=answer,  # type: ignore
            max_comparisons=int(max_comparisons),
            uuid=uuid,
            suffix=suffix,
        )

    def load_qs_dataset(self) -> "QsDataset":
        qsds = QsDataset.from_yaml_file(self.qs_dataset_path)
        assert isinstance(qsds, QsDataset)
        return qsds

    def load_direct_eval(self, instr_id: str, model_id: str) -> "DirectEval":
        direct_eval = DirectEval.from_yaml_file(
            self.direct_eval_path(instr_id, model_id)
        )
        assert isinstance(direct_eval, DirectEval)
        return direct_eval

    def load_cot_eval(
        self, instr_id: str, model_id: str, sampling_params: SamplingParams
    ) -> "CotEval":
        cot_eval = CotEval.from_yaml_file(
            self.cot_eval_path(instr_id, model_id, sampling_params)
        )
        assert isinstance(cot_eval, CotEval)
        return cot_eval

    def load_old_cot_eval(
        self, instr_id: str, model_id: str, sampling_params: SamplingParams
    ) -> "OldCotEval":
        cot_eval = OldCotEval.from_yaml_file(
            self.cot_eval_path(instr_id, model_id, sampling_params)
        )
        assert isinstance(cot_eval, OldCotEval)
        return cot_eval


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(DatasetParams)
DumpMeta(key_transform="SNAKE").bind_to(DatasetParams)


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
class AtCoderQuestion(YAMLWizard):
    """Represents a single AtCoder problem with its solution attempt."""

    name: str  # contest_id_problem_id
    problem: str  # Combined prompt with all problem info
    cpp_solution: str | None = None  # The C++ solution if available


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderQuestion)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderQuestion)


@dataclass
class AtCoderDatasetParams(YAMLWizard):
    """Parameters for an AtCoder dataset."""

    description: str
    id: str | None
    pre_id: str | None
    dataset_type: Literal["atcoder"] = "atcoder"


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderDatasetParams)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderDatasetParams)


@dataclass
class AtCoderResponse(AtCoderQuestion):
    """A response to an AtCoder problem."""

    model_answer: list[str] | list[StepFaithfulness] | None = (
        None  # The C++ solution, wrapped in tags
    )
    model_thinking: str | None = None  # The step-by-step reasoning

    correctness_explanation: str | None = None
    correctness_is_correct: bool | None = None
    correctness_classification: (
        Literal["EQUIVALENT", "NOT_EQUIVALENT", "NA_NEITHER", "NA_BOTH"] | None
    ) = None
    cpp_solution: str | None = None


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderResponse)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderResponse)


@dataclass
class AtCoderStats(YAMLWizard):
    """Statistics for AtCoder submissions."""

    finding_code_failed: int = -1
    compilation_failed: int = -1
    atcodertools_cmd_failed: int = -1
    solution_failed: int = -1
    solution_passed: int = -1

    @classmethod
    def zeros(cls) -> "AtCoderStats":
        return cls(
            finding_code_failed=0,
            compilation_failed=0,
            atcodertools_cmd_failed=0,
            solution_failed=0,
            solution_passed=0,
        )


LoadMeta(
    v1=True, v1_unsafe_parse_dataclass_in_union=True, key_transform="SNAKE"
).bind_to(AtCoderStats)
DumpMeta(key_transform="SNAKE").bind_to(AtCoderStats)


@dataclass
class CotResponses(YAMLWizard):
    # For normal datasets
    # qid -> {uuid -> response_str or a MathResponse or AtCoderResponse}
    responses_by_qid: dict[str, dict[str, MathResponse | AtCoderResponse | str]]
    model_id: str
    instr_id: str
    ds_params: AtCoderDatasetParams | MathDatasetParams | DatasetParams
    sampling_params: SamplingParams | DefaultSamplingParams

    # Fields for AtCoder submission tracking:
    atcoder_stats: AtCoderStats = field(default_factory=AtCoderStats)

    def get_path(self, suffix: str = "") -> Path:
        if isinstance(self.ds_params, DatasetParams) and isinstance(
            self.sampling_params, SamplingParams
        ):
            return self.ds_params.cot_responses_path(
                instr_id=self.instr_id,
                model_id=self.model_id,
                sampling_params=self.sampling_params,
            )

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
    split_responses_by_qid: dict[str, dict[str, MathResponse | AtCoderResponse]]

    successfully_split_count: int
    failed_to_split_count: int
    instr_id: str
    ds_params: AtCoderDatasetParams | MathDatasetParams
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
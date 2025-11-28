import json
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizer


class HFQADataset(IterableDataset):
    """Iterable HuggingFace QA dataset wrapper with on-the-fly tokenization.

    Args:
        dataset_name: HuggingFace dataset hub name (e.g., "locuslab/TOFU").
        subset_name: Dataset configuration name (e.g., "retain_perturbed").
        split: Dataset split (e.g., "train").
        tokenizer: PreTrainedTokenizer used for tokenization.
        question_key: Field name for question.
        answer_key: Field name for answer.
        max_length: Max sequence length for tokenization.

    Returns:
        Iterable over tokenized samples with keys: "input_ids", "attention_mask".

    Raises:
        ValueError: If dataset fields are missing.
    """

    def __init__(
        self,
        dataset_name: str,
        subset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        question_key: str = "question",
        answer_key: str = "answer",
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.split = split
        self.tokenizer = tokenizer
        self.question_key = question_key
        self.answer_key = answer_key
        self.max_length = max_length

        try:
            self.ds = load_dataset(self.dataset_name, name=self.subset_name, split=self.split, streaming=True)
        except Exception as e:
            raise ValueError(f"Failed to load dataset {dataset_name} (name={subset_name}, split={split}). Error: {e}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for row in self.ds:  # streaming iterator
            if self.question_key not in row:
                raise ValueError(f"Missing question_key '{self.question_key}' in dataset row.")
            if self.answer_key not in row:
                raise ValueError(f"Missing answer_key '{self.answer_key}' in dataset row.")

            q = row[self.question_key] or ""
            a = row[self.answer_key] or ""
            text = f"Question: {q}\nAnswer: {a}"

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            yield {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }


class MergedIterableDataset(IterableDataset):
    """Merge multiple IterableDatasets sequentially.

    Args:
        datasets: List of IterableDataset instances to merge.

    Returns:
        Iterable yielding samples from each dataset in order.
    """

    def __init__(self, datasets: List[IterableDataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for ds in self.datasets:
            for item in ds:
                yield item


def _build_hf_dataset(
    tokenizer: PreTrainedTokenizer,
    subset_name: str,
    split: str,
    question_key: str,
    answer_key: str,
    dataset_name: str = "locuslab/TOFU",
    max_length: int = 512,
) -> HFQADataset:
    return HFQADataset(
        dataset_name=dataset_name,
        subset_name=subset_name,
        split=split,
        tokenizer=tokenizer,
        question_key=question_key,
        answer_key=answer_key,
        max_length=max_length,
    )


def get_loaders(
    tokenizer: PreTrainedTokenizer,
    forget_name: str,
    retain_names: List[str],
    forget_split: str = "train",
    retain_split: str = "train",
    question_key: str = "question",
    forget_answer_key: str = "answer",
    retain_answer_key: Optional[str] = None,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Construct DataLoaders for Forget and Retain sets using HuggingFace streaming datasets.

    Args:
        tokenizer: Tokenizer for Llama‑3.1.
        forget_name: TOFU subset name for forget set.
        retain_names: List of TOFU subset names for retain sets.
        forget_split: Split for forget dataset.
        retain_split: Split for retain datasets.
        question_key: Field name for question.
        forget_answer_key: Field name for answer in forget dataset.
        retain_answer_key: Global answer field for retain datasets; if None and a name equals
            "retain_perturbed", use "perturbed_answer".
        batch_size: Batch size.
        max_length: Max tokenization length.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (forget_loader, retain_loader).
    """

    forget_ds = _build_hf_dataset(
        tokenizer=tokenizer,
        subset_name=forget_name,
        split=forget_split,
        question_key=question_key,
        answer_key=forget_answer_key,
        max_length=max_length,
    )

    retain_datasets: List[IterableDataset] = []
    for name in retain_names:
        ans_key = retain_answer_key
        if ans_key is None and name == "retain_perturbed":
            ans_key = "perturbed_answer"
        if ans_key is None:
            ans_key = "answer"
        try:
            ds = _build_hf_dataset(
                tokenizer=tokenizer,
                subset_name=name,
                split=retain_split,
                question_key=question_key,
                answer_key=ans_key,
                max_length=max_length,
            )
            retain_datasets.append(ds)
        except Exception as e:
            print(f"Warning: failed to load retain dataset '{name}'. Error: {e}")

    if not retain_datasets:
        raise ValueError("No retain datasets loaded!")

    retain_merged = MergedIterableDataset(retain_datasets)

    forget_loader = DataLoader(
        forget_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    retain_loader = DataLoader(
        retain_merged,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return forget_loader, retain_loader


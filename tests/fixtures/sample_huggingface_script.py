"""Minimal HuggingFace Datasets script for format detection tests."""
from datasets import load_dataset

dataset = load_dataset("imdb")
train_data = dataset["train"]

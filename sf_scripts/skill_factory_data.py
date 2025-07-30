import json

import os
import argparse

from datasets import load_dataset, concatenate_datasets
from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="TAUR-dev/D-DATA-canonical_dataset_splits-v1-7_13_25")

    parser.add_argument('--task_configs', default=['countdown_3arg'], type=str, nargs='+', help="The task config to filter the dataset")
    parser.add_argument('--training_splits', default=['rl_train'], nargs='+')

    parser.add_argument("--validation_splits", default=['val'], nargs='+', help="The splits to load from the dataset")

    parser.add_argument("--output_folder", type=str)
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    training_datasets = []
    validation_datasets = []

    for task_config in args.task_configs:
        dataset = load_dataset(args.hf_repo, task_config)

        for split in args.training_splits:
            if split not in dataset:
                raise ValueError(f"Split {split} not found in dataset {args.hf_repo} for task config {task_config}")

            training_datasets.append(dataset[split])

        for split in args.validation_splits:
            if split not in dataset:
                raise ValueError(f"Split {split} not found in dataset {args.hf_repo} for task config {task_config}")

            validation_datasets.append(dataset[split])

    train_dataset = concatenate_datasets(training_datasets)
    test_dataset = concatenate_datasets(validation_datasets)

    # shuffle
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)

    if args.limit:
        train_dataset = train_dataset.select(range(min(args.limit, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            extra_cols = json.loads(example['all_other_columns'])

            data = {
                "data_source": example["task_source"],
                "prompt": example['prompt'],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": example['answer']},
                "extra_info": {
                    **example,
                    **extra_cols
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.output_folder
    hdfs_dir = args.hdfs_dir


    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

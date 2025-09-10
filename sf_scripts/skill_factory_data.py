import json

import os
import argparse

from datasets import load_dataset, concatenate_datasets
from verl.utils.hdfs_io import copy, makedirs



VERBOSE_TEMPLATE = """Answer the following problem. Explain your reasoning step by step. At the very end, give your answer in this format: <answer>(your answer)</answer>.

# Required Format
Your response must follow this exact structure with the specified XML tags. Specifically, your answer should have the following format with the correct tags (e.g., <think>), where you make a number of repeated samples/attempts at solving the question, reflect on each attempt with "correct" or "incorrect" verdicts, and end up choosing the final answer based on your attempts and reflections. Each of your attempts can incorporate what you learned from past attempt(s) and reflection(s).
<think>
    <sample>
        [Your detailed reasoning and step-by-step solution approach for this attempt]
        <answer>[Your answer for this attempt]</answer>
    </sample>
    <reflect>
        [Analyze your approach: What worked? What didn't? What should you do differently?]
        <verdict>[Either "correct" or "incorrect" - be honest about your attempt]</verdict>
    </reflect>
    <sample>
        [Your improved reasoning and solution approach, incorporating lessons from previous attempts]
        <answer>[Your answer for this attempt]</answer>
    </sample>
    <reflect>
        [Analyze this improved approach and its correctness]
        <verdict>[Either "correct" or "incorrect" - be honest about your attempt]</verdict>
    </reflect>
    [Continue with additional samples and reflections as needed...]
</think>
[Your final analysis and reasoning based on all your attempts and reflections]
<answer>[Your final, most confident answer]</answer>

# Example Response
Let's say you are asked to multiply 37 by 24, your response could look like the example below. While the actual problem may not be multiplication and may not even be mathematical, you should learn from this example to understand how to create high-quality attempts and reflections.
<think>
    <sample>
        I'll use long multiplication to compute 37 x 24:
        - Step 1: First multiply by 4:
          37 x 4 = (30 x 4) + (7 x 4) = 120 + 28 = 148.
        - Step 2: Next multiply by 20:
          37 x 20 = 740.
        - Step 3: Add partials:
          148 + 740 = 868
        <answer>868</answer>
    </sample>
    <reflect>
        Let's check this attempt step by step:
        - Step 1: 30 x 4 is indeed 120, and same for 7 x 4 being 28. 120 + 28 is 148. This step is correct.
        - Step 2: 37 x 2 is 74. Since 20 is 10 times 2, 37 x 20 is therefore 740. this step is also correct.
        - Step 3: by adding 148 and 740 digit by digit, I realize that their middle digit sums to 8 and there is no carry from the last digit, this is different from the second digit of 868. This step is wrong.
        I added 148 + 740 incorrectly. I must have mis-added the tens column previously.
        <verdict>incorrect</verdict>
    </reflect>
    <sample>
        Based on my previous attempts and reflections, I notice that my first attempt's mistake is in adding 148 + 740 incorrectly. Let's do it carefully this time:
        - starting from the last digit, we have 8 + 0 = 8, no carry.
        - next at the second digit, we have 4 + 4 = 8, no carry.
        - next at the first digit, we have 1 + 7 = 8, no carry.
        Therefore, the final answer should be 888.
        <answer>888</answer>
    </sample>
    <reflect>
        The new attempt tries to correct the mistake in the last step of the first attempt. It breaks down 148 + 740 into digit-wise additions, adds them correctly, and puts them together into the final number. Therefore, I think the new attempt is correct.
        <verdict>correct</verdict>
    </reflect>
    <sample>
        Based on my previous attempts and reflections, it seems like the second attempt fixes the error from the first attempt. However, I should double check that it is actually correct.
        37 x 24
        = (30 + 7) x (20 + 4)
        = 30 x 20 + 30 x 4 + 7 x 20 + 7 x 4
        = 600 + 120 + 140 + 28
        = 720 + 140 + 28
        = 860 + 28
        = 888
        <answer>888</answer>
    </sample>
    <reflect>
        The new attempt tries to do the calculations again and arrives at the same answer as my second attempt. However, I should always reflect on the new attempt to catch any potential errors.
        - the first line breaks 37 into 30 + 7 and 24 into 20 + 4, this is correct.
        - the second line uses distributive property to expand the previous expression, this is correct.
        - the third line performs 4 multiplications, 30 x 20 is indeed 600, 30 x 4 is indeed 120, 7 x 20 is indeed 140, and 7 x 4 is indeed 28. This is correct.
        - the fourth line adds 600 with 120 to get 720. This is correct.
        - the fifth line adds 720 with 140 to get 860. This is correct.
        - the sixth line adds 860 with 28 to get 888. This is correct.
        <verdict>correct</verdict>
    </reflect>
</think>
The first attempt mis-added 148 + 740 as 868; the reflection isolated the tens-column addition error. The second attempt fixed this error from the first attempt, yielding 888. The third attempt carefully recalculates everything again line by line, yielding 888 again.
<answer>888</answer>

$PROBLEM

# Important Guidelines
- Be thorough in your reasoning for each attempt
- Show all mathematical steps clearly
- Be honest in your self-assessment during reflections
- If you're unsure about a step, explain your uncertainty
- Use multiple attempts to verify your work, as many as it needs for you to be very confident about the answer
- Double-check arithmetic and algebraic manipulations
- Consider alternative approaches if your past attempt(s) fail
- Make sure to give your final answer in this format: <answer>[your answer]</answer>.
- Let's think step by step, do not jump to conclusions in any attempt or reflection."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="TAUR-dev/D-DATA-canonical_dataset_splits-v1-7_13_25")
    parser.add_argument("--verbose_prompt", action='store_true')

    parser.add_argument('--task_configs', default=['countdown_3arg'], type=str, nargs='+', help="The task configs to use for training data")
    parser.add_argument('--validation_task_configs', default=None, type=str, nargs='+', help="The task configs to use for validation data (defaults to task_configs if not specified)")
    parser.add_argument('--training_splits', default=['rl_train'], nargs='+')

    parser.add_argument("--validation_splits", default=['val'], nargs='+', help="The splits to load from the dataset")

    parser.add_argument("--output_folder", type=str)
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # If validation_task_configs not specified, use task_configs
    if args.validation_task_configs is None:
        args.validation_task_configs = args.task_configs

    training_datasets = []
    validation_datasets = []

    # Process training tasks
    for task_config in args.task_configs:
        dataset = load_dataset(args.hf_repo, task_config)

        for split in args.training_splits:
            if split not in dataset:
                raise ValueError(f"Split {split} not found in dataset {args.hf_repo} for task config {task_config}")

            training_datasets.append(dataset[split])

    # Process validation tasks
    for task_config in args.validation_task_configs:
        dataset = load_dataset(args.hf_repo, task_config)

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
                "data_source": example["task_config"],
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

    # use the long prompt
    if args.verbose_prompt:
        def add_verbose_prompt(example):
            assert len(example['prompt']) == 1
            assert example['prompt'][0]['role'] == 'user'
            content = example['prompt'][0]['content']
            problem_start_idx = content.find("# Problem")
            problem_end_idx = content.find("Give your answer in the following format:")
            assert 0 <= problem_start_idx < problem_end_idx
            # replace the prompt with template
            example['prompt'][0]['content'] = VERBOSE_TEMPLATE.replace("$PROBLEM", content[problem_start_idx: problem_end_idx].strip())
            return example

        train_dataset = train_dataset.map(function=add_verbose_prompt)
        test_dataset = test_dataset.map(function=add_verbose_prompt)

    local_dir = args.output_folder
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

from skill_factory.data.sources.clean_evaluation_helpers import evaluate_dataset
from skill_factory.data.sources.answer_extraction import TagPattern, AnswerExtractor

from datasets import Dataset
import json
import re
from typing import List, Union, Optional, Dict
from difflib import SequenceMatcher


def compute_transition_penalty(internal_answers_is_correct: List[bool]) -> float:
    """Count number of True->False Transitions, divide by number of transitions to normalize to [0, 1].
       Includes the transition from the last sample to the final answer.
    """
    if len(internal_answers_is_correct) < 2:
        return 0.0

    transition_penalty = 0.0
    for i in range(len(internal_answers_is_correct) - 1):
        if internal_answers_is_correct[i] and not internal_answers_is_correct[i + 1]:
            transition_penalty += 1.0
    transition_penalty /= (len(internal_answers_is_correct) - 1)
    assert 0 <= transition_penalty <= 1.0

    return transition_penalty


def compute_simple_format_reward(solution_str: str) -> float:
    """Return 1 if solution_str follows <think>[any tags]</think><answer></answer>, else 0.0."""
    # Find all tags in the solution_str
    tag_pattern = r'<(/?)(?:think|answer)>'
    matches = list(re.finditer(tag_pattern, solution_str))
    matches = [match.group(0) for match in matches]

    if len(matches) < 4:
        return 0.0
    if matches[0] == '<think>' and matches[-3:] == ['</think>', '<answer>', '</answer>']:
        return 1.0
    return 0.0


def compute_complex_format_reward(solution_str: str, response_or_sample: str) -> float:
    """
        Return 1 if a string follows
        <think> <sample><answer></answer></sample><reflect><verdict></verdict></reflect> </think><answer></answer>
        where the middle part can be repeated >=1 time(s),
        else 0.0.
    """
    # Find all tags in the solution_str
    tag_pattern = fr'<(/?)(?:think|{response_or_sample}|reflect|verdict|answer)>'
    matches = list(re.finditer(tag_pattern, solution_str))
    matches = [match.group(0) for match in matches]

    # if should at least pass simple_format_reward
    if len(matches) < 4:
        return 0.0
    if matches[0] != '<think>' or matches[-3:] != ['</think>', '<answer>', '</answer>']:
        return 0.0

    # check if the middle part is a non-empty repetition of the template
    template = [
        f'<{response_or_sample}>',
            '<answer>',
            '</answer>',
        f'</{response_or_sample}>',
        '<reflect>',
            '<verdict>',
            '</verdict>',
        '</reflect>'
    ]
    matches_within_think_tags = matches[1:-3]
    if len(matches_within_think_tags) == 0:
        return 0.0

    while len(matches_within_think_tags) > 0:
        if matches_within_think_tags[:len(template)] != template:
            return 0.0
        matches_within_think_tags = matches_within_think_tags[len(template):]

    return 1.0


def extract_tag_text(text: str, tag: str) -> List[str]:
    """Extract all text between <tag> and </tag>."""
    pattern = fr'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    return [s.strip() for s in matches if s.strip()]


def remove_answer_tags(text: str) -> str:
    """Remove all text between <answer> and </answer> tags from the input text."""
    pattern = r'<answer>.*?</answer>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def compute_lexical_similarity(text1: str, text2: str) -> float:
    """Compute lexical similarity between two texts using SequenceMatcher."""
    # Normalize whitespace for better comparison
    text1 = ' '.join(text1.split())
    text2 = ' '.join(text2.split())

    similarity = SequenceMatcher(None, text1, text2).ratio()
    assert 0 <= similarity <= 1.0
    return similarity


def compute_similarity_penalty(
    solution_str: str,
    response_or_sample: str,
    threshold: float = 0.9,
) -> float:
    """
    Compute similarity penalty based on lexical similarity between samples.

    Extracts text between <sample></sample> tags, removes content between
    <answer></answer> tags, then calculates lexical similarity between samples.
    If similarity is greater than the threshold, returns 1.0, else 0.0.

    Args:
        solution_str: Model's solution string
        threshold: Similarity threshold above which penalty is applied

    Returns:
        1.0 if similarity exceeds threshold (penalty should be applied), 0.0 otherwise
    """
    # Extract sample texts
    samples = extract_tag_text(solution_str, tag=response_or_sample)

    # Need at least 2 samples to compare
    if len(samples) < 2:
        return 0.0

    # Remove answer tags from each sample
    cleaned_samples = [remove_answer_tags(sample) for sample in samples]

    # Check similarity between all pairs of samples
    for i in range(len(cleaned_samples)):
        for j in range(i + 1, len(cleaned_samples)):
            similarity = compute_lexical_similarity(cleaned_samples[i], cleaned_samples[j])
            if similarity > threshold:
                return 1.0

    return 0.0


def compute_sample_correctness_reward(
    solution_str: str,
    response_or_sample: str,
    internal_answers_is_correct: List[bool],
) -> float:
    """
    Compute reward based on correctness of sample tags.

    Takes the first N booleans from internal_answers_is_correct where N equals
    the number of sample tags found in the response. Returns # correct / N.

    Args:
        solution_str: Model's solution string
        internal_answers_is_correct: List of booleans indicating correctness of internal answers

    Returns:
        Ratio of correct samples (# correct / N), max value is 1.0
    """
    if len(internal_answers_is_correct) == 0:
        return 0.0

    # Extract sample texts to count them
    samples = extract_tag_text(solution_str, tag=response_or_sample)
    num_samples = len(samples)

    # If no samples found, return 0
    if num_samples == 0:
        return 0.0

    # If we don't have enough internal answers, return 0
    if len(internal_answers_is_correct) < num_samples:
        return 0.0

    # Take first N booleans where N = number of sample tags
    relevant_answers = internal_answers_is_correct[:num_samples]

    # Calculate ratio of correct answers
    num_correct = sum(relevant_answers)
    return num_correct / num_samples


def compute_sample_count_penalty(solution_str: str, response_or_sample: str, sample_count: int) -> float:
    """Return 1.0 if there is fewer than sample_count"""
    samples = extract_tag_text(solution_str, tag=response_or_sample)
    return int(len(samples) < sample_count)


def compute_verdict_correctness_reward(solution_str: str, internal_answers_is_correct: List[bool], tag: str) -> float:
    """Compute average verification correctness"""
    verdict_correctness_reward = 0.0
    verdicts = extract_tag_text(solution_str, tag)

    if len(verdicts) > 0 and len(internal_answers_is_correct) == len(verdicts) + 1:
        # valid format, now test verdicts
        for c, v in zip(internal_answers_is_correct[:-1], verdicts):
            verdict_is_correct = ('wrong' not in v.lower()) and ('incorrect' not in v.lower()) and \
                ('not correct' not in v.lower())
            if c == verdict_is_correct:
                verdict_correctness_reward += 1.0
        return verdict_correctness_reward / len(verdicts)

    return 0.0


def compute_final_answer_in_samples_reward(internal_answers: List[str]) -> float:
    """Check if the final answer is equal to any of the sample answers before it"""
    if len(internal_answers) > 1:
        for sample_idx in range(len(internal_answers) - 1):
            if internal_answers[sample_idx].strip() == internal_answers[-1].strip():
                return 1.0
    return 0.0


def compute_score_batch(
        data_sources: List[str],
        solution_strs: List[str],
        ground_truths: List[str],
        # config
        response_or_sample: str,
        # extra reward weights (see defaults in ppo_trainer.yaml)
        simple_format_reward_weight: float,
        complex_format_reward_weight: float,
        sample_correctness_reward_weight: float,
        verdict_correctness_reward_weight: float,
        reflection_correctness_reward_weight: float,
        final_answer_in_samples_reward_weight: float,
        # extra penalty weights (see defaults in ppo_trainer.yaml)
        transition_penalty_weight: float,
        similarity_penalty_weight: float,
        sample_count_penalty_weight: float,
        # reward clipping (see defaults in ppo_trainer.yaml)
        reward_min: float,
        reward_max: float,
        extra_infos: Optional[List[Union[str, dict]]] = None,
    ) -> List[Dict[str, float]]:
    """
    Batched version of compute_score for much better performance.

    Args:
        data_sources: List of data source names
        solution_strs: List of model solution strings
        ground_truths: List of correct answers

        response_or_sample: either "response" or "sample", based on which SFT dataset we use

        simple_format_reward_weight: whether think and answer open-and-close tags are present
        complex_format_reward_weight: whether model properly thinks with <sample>, <answer>, <reflect>, and <verdict>
        sample_correctness_reward_weight: average correctness of samples (not including the final answer)
        verdict_correctness_reward_weight: average correctness in whether the model self-judges correctly, extracted from verdict tags
        reflection_correctness_reward_weight: average correctness in whether the model self-judges correctly, extracted from reflection tags
        final_answer_in_samples_reward_weight: whether the final answer is one of the sampled answers

        transition_penalty_weight: number of incorrect-to-correct occurances, including from the last sample to the final answer
        similarity_penalty_weight: penalty for when sample content (excluding answer) is too similar
        sample_count_penalty_weight: penalty for when there are <2 samples

        reward_min: minimum reward value
        reward_max: maximum reward value
        extra_infos: List of extra info dicts

    Returns:
        List of score dictionaries, one per input sample
    """
    batch_size = len(solution_strs)
    if batch_size == 0:
        return []

    assert len(data_sources) == batch_size
    assert len(ground_truths) == batch_size
    assert len(extra_infos) == batch_size

    # Prepare batch data for evaluation
    dataset_rows = []
    for i in range(batch_size):
        extra_info = extra_infos[i] or {}
        if not isinstance(extra_info, dict):
            extra_info = json.loads(extra_info)
        assert 'task_source' in extra_info, f"extra_info must contain 'task_source' key for sample {i}"

        dataset_rows.append({
            **extra_info,
            'model_responses__verl': [solution_strs[i]],
            'answer': ground_truths[i],
            # 'task_source': data_sources[i], # task_config is passed here from skill_factory_data, not task_source
            'task_source': extra_info['task_source'],
        })


    # evaluate model response
    ds = Dataset.from_list(dataset_rows)
    answer_tag_pattern = TagPattern('<answer>', '</answer>', 'answer tag', 1.0)
    extractor = AnswerExtractor([answer_tag_pattern], [answer_tag_pattern])
    result = evaluate_dataset(ds, ['model_responses__verl'], verbose=False, custom_extractor=extractor)

    # all intermediate answers that are in <answer> tags (including the final answer so you may want to ignore the last one.)
    # internal_answers = result['model_responses__verl__internal_answers__eval_extracted_answers']

    # store column queries to only access once
    final_is_correct_list = result['model_responses__verl__eval_is_correct']
    internal_answers_is_correct_list = result['model_responses__verl__internal_answers__eval_is_correct']
    internal_answers_list = result['model_responses__verl__internal_answers__eval_extracted_answers']
    assert len(final_is_correct_list) == len(internal_answers_is_correct_list) == len(internal_answers_list) == batch_size


    # quick checks for format because im paranoid
    assert all(len(x) == 1 for x in final_is_correct_list) # one solution str
    assert all(len(x) == 1 for x in internal_answers_is_correct_list if x != None) # one solution str
    assert all(len(x) == 1 for x in internal_answers_list if x != None) # one solution str

    assert [i for i, x in enumerate(internal_answers_is_correct_list) if x == None] == \
         [i for i, x in enumerate(internal_answers_list) if x == None]
    assert [len(x[0]) for x in internal_answers_is_correct_list if x != None] == \
        [len(x[0]) for x in internal_answers_list if x != None]

    assert all(x[0] in [True, False] for x in final_is_correct_list)
    assert all(set(x[0]).issubset({True, False}) for x in internal_answers_is_correct_list if x != None)


    # compute and store a bunch of info in score
    scores = []
    for solution_str, final_is_correct, internal_answers_is_correct, internal_answers in \
        zip(solution_strs, final_is_correct_list, internal_answers_is_correct_list, internal_answers_list):

        internal_answers = internal_answers[0] if internal_answers != None else []
        internal_answers_is_correct = internal_answers_is_correct[0] if internal_answers_is_correct != None else []
        final_answer_is_correct = float(final_is_correct[0])

        # more assertions that number of internal answers equals number of answer tags
        assert len(extract_tag_text(solution_str, tag='answer')) == len(internal_answers) == len(internal_answers_is_correct)
        if final_answer_is_correct:
            assert len(internal_answers_is_correct) > 0 and internal_answers_is_correct[-1]
        else:
            assert len(internal_answers_is_correct) == 0 or not internal_answers_is_correct[-1]

        # rewards
        simple_format_reward = compute_simple_format_reward(solution_str)
        complex_format_reward = compute_complex_format_reward(solution_str, response_or_sample)
        sample_correctness_reward = compute_sample_correctness_reward(
            solution_str=solution_str,
            internal_answers_is_correct=internal_answers_is_correct,
            response_or_sample=response_or_sample,
        )
        verdict_correctness_reward = compute_verdict_correctness_reward(
            solution_str=solution_str,
            internal_answers_is_correct=internal_answers_is_correct,
            tag='verdict',
        )
        reflection_correctness_reward = compute_verdict_correctness_reward(
            solution_str=solution_str,
            internal_answers_is_correct=internal_answers_is_correct,
            tag='reflect',
        )
        final_answer_in_samples_reward = compute_final_answer_in_samples_reward(internal_answers=internal_answers)

        # penalties
        transition_penalty = compute_transition_penalty(internal_answers_is_correct)
        similarity_penalty = compute_similarity_penalty(solution_str, response_or_sample)
        sample_count_penalty = compute_sample_count_penalty(solution_str, response_or_sample, sample_count=2)

        # Combine rewards
        combined_reward = final_answer_is_correct \
            + simple_format_reward * simple_format_reward_weight \
            + complex_format_reward * complex_format_reward_weight \
            + sample_correctness_reward * sample_correctness_reward_weight \
            + verdict_correctness_reward * verdict_correctness_reward_weight \
            + reflection_correctness_reward * reflection_correctness_reward_weight \
            + final_answer_in_samples_reward * final_answer_in_samples_reward_weight \
            - transition_penalty * transition_penalty_weight \
            - similarity_penalty * similarity_penalty_weight \
            - sample_count_penalty * sample_count_penalty_weight

        # Clip reward to bounds
        combined_reward = max(reward_min, min(reward_max, combined_reward))

        # is any answer correct
        is_any_answer_correct = int(any(internal_answers_is_correct))

        # Return both the final reward and the individual components for logging
        scores.append({
            # reward for training
            "score": combined_reward,
            # rewards for logging
            "final_answer_is_correct": final_answer_is_correct,
            "simple_format_reward": simple_format_reward,
            "complex_format_reward": complex_format_reward,
            "sample_correctness_reward": sample_correctness_reward,
            "verdict_correctness_reward": verdict_correctness_reward,
            "reflection_correctness_reward": reflection_correctness_reward,
            "final_answer_in_samples_reward": final_answer_in_samples_reward,
            # penalties for logging
            "transition_penalty": transition_penalty,
            "similarity_penalty": similarity_penalty,
            "sample_count_penalty": sample_count_penalty,
            # just for logging
            "num_sample_tag": len(extract_tag_text(solution_str, tag=response_or_sample)),
            "num_verdict_tag": len(extract_tag_text(solution_str, tag='verdict')),
            "num_reflect_tag": len(extract_tag_text(solution_str, tag='reflect')),
            "num_think_tag": len(extract_tag_text(solution_str, tag='think')),
            "num_answer_tag": len(extract_tag_text(solution_str, tag='answer')),
            "is_any_answer_correct": is_any_answer_correct,
        })

        # another sanity check
        if complex_format_reward:
            assert simple_format_reward
        assert is_any_answer_correct >= final_answer_is_correct

    return scores

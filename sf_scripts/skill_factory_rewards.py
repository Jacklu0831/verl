from skill_factory.data.sources.clean_evaluation_helpers import evaluate_dataset
from skill_factory.data.sources.answer_extraction import TagPattern, AnswerExtractor

from datasets import Dataset
import json
import re
from typing import List, Union, Optional
from difflib import SequenceMatcher


def compute_transition_penalty(all_intermediate_is_correct: List[bool]) -> float:
    """
    Count number of True->False Transitions, divide by number of transitions to normalize to [0, 1].
    """
    if len(all_intermediate_is_correct) < 2:
        return 0.0

    transition_penalty = 0.0
    for i in range(len(all_intermediate_is_correct) - 1):
        if all_intermediate_is_correct[i] and not all_intermediate_is_correct[i + 1]:
            transition_penalty += 1.0
    transition_penalty /= (len(all_intermediate_is_correct) - 1)
    assert 0 <= transition_penalty <= 1.0

    return transition_penalty


def compute_format_score(solution_str: str) -> float:
    """
    Compute formatting score based on proper think and answer tag pairs.

    Returns 1.0 if all satisfied:
    - At least one <think></think> pair exists
    - At least one <answer></answer> pair exists
    - All opening tags have corresponding closing tags
    - Tags appear in correct order (<think> before </think>, <answer> before </answer>)

    Returns 0.0 otherwise.
    """
    # Find all tag positions
    think_open_pos = [m.start() for m in re.finditer(r'<think>', solution_str)]
    think_close_pos = [m.start() for m in re.finditer(r'</think>', solution_str)]
    answer_open_pos = [m.start() for m in re.finditer(r'<answer>', solution_str)]
    answer_close_pos = [m.start() for m in re.finditer(r'</answer>', solution_str)]

    # has think and answer pair
    has_pair = len(think_open_pos) >= 1 and len(think_close_pos) >= 1 and len(answer_open_pos) >= 1 and len(answer_close_pos) >= 1
    if not has_pair:
        return 0.0

    # is balanced
    is_balanced = (len(think_open_pos) == len(think_close_pos)) and (len(answer_open_pos) == len(answer_close_pos))
    if not is_balanced:
        return 0.0

    # is ordered
    is_ordered = all(open < close for open, close in zip(think_open_pos, think_close_pos)) and \
        all(open < close for open, close in zip(answer_open_pos, answer_close_pos))

    return float(is_ordered)


def compute_format_score_v2(solution_str: str) -> float:
    """
    Return 1 if a string follows the exact format specified, else 0.

    <think>
        <sample>
        {trace 1, can be correct or incorrect, include an <answer>...</answer>}
        </sample>
        <reflect>
        {simple static glue phrase}
        </reflect>
        <sample>
        {trace 2, always correct, include an <answer>...</answer>}}
        </sample>
    </think>
    <answer>
    {gold answer}
    </answer>
    """
    # Find all tags in the solution_str
    tag_pattern = r'<(/?)(?:think|sample|reflect|answer)>'
    matches = list(re.finditer(tag_pattern, solution_str))

    # Extract tags in order
    found_tags = [match.group(0) for match in matches]

    # Right number of tags and in the correct order
    if found_tags != [
        '<think>',
            '<sample>',
                '<answer>',
                '</answer>',
            '</sample>',
            '<reflect>',
            '</reflect>',
            '<sample>',
                '<answer>',
                '</answer>',
            '</sample>',
        '</think>',
        '<answer>',
        '</answer>'
    ]:
        return 0.0

    # Check that the reflect section contains the exact phrase
    reflect_start = None
    reflect_end = None
    for match in matches:
        if match.group(0) == '<reflect>':
            reflect_start = match.end()
        elif match.group(0) == '</reflect>':
            reflect_end = match.start()
            break

    if reflect_start is None or reflect_end is None:
        return 0.0

    reflect_content = solution_str[reflect_start:reflect_end].strip()
    if reflect_content != "Actually, wait, let me be more careful about this.":
        return 0.0

    return 1.0


def extract_sample_text(text: str) -> List[str]:
    """
    Extract all text between <sample> and </sample> tags.
    
    Args:
        text: Input text containing sample tags
        
    Returns:
        List of strings containing the content between sample tags
    """
    pattern = r'<sample>(.*?)</sample>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def remove_answer_tags(text: str) -> str:
    """
    Remove all text between <answer> and </answer> tags from the input text.
    
    Args:
        text: Input text that may contain answer tags
        
    Returns:
        Text with answer tag content removed
    """
    pattern = r'<answer>.*?</answer>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def compute_lexical_similarity(text1: str, text2: str) -> float:
    """
    Compute lexical similarity between two texts using SequenceMatcher.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Similarity ratio between 0 and 1
    """
    # Normalize whitespace for better comparison
    text1 = ' '.join(text1.split())
    text2 = ' '.join(text2.split())
    
    return SequenceMatcher(None, text1, text2).ratio()


def compute_similarity_penalty(solution_str: str, threshold: float = 0.9) -> float:
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
    samples = extract_sample_text(solution_str)
    
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


def compute_sample_correctness_reward(solution_str: str, internal_answers_is_correct: List[bool]) -> float:
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
    # Extract sample texts to count them
    samples = extract_sample_text(solution_str)
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


def compute_sample_count_penalty(solution_str: str) -> float:
    """
    Compute penalty based on number of sample tags.
    
    Returns -1 penalty if 0 or 1 sample tags are found, 0 otherwise.
    
    Args:
        solution_str: Model's solution string
        
    Returns:
        -1.0 if 0 or 1 sample tags found, 0.0 if 2 or more sample tags
    """
    # Extract sample texts to count them
    samples = extract_sample_text(solution_str)
    num_samples = len(samples)
    
    # Apply penalty if 0 or 1 sample tags
    if num_samples < 2:
        return 1.0
    
    return 0.0


def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        format_score_weight: float,
        format_score_v2_weight: float,
        transition_penalty_weight: float,
        similarity_penalty_weight: float,
        sample_correctness_weight: float,
        sample_count_penalty_weight: float,
        reward_min: float,
        reward_max: float,
        extra_info: Optional[Union[str, dict]] = None
    ):
    """
    Compute reward score

    Args:
        data_source: Name of the data source for task-specific evaluation scheme (e.g., countdown)
        solution_str: Model's solution string
        ground_truth: Correct answer (e.g., 5 + 87 + 73)
        format_score_weight: weight for formatting reward, which is added to other rewards
        format_score_v2_weight: weight for formatting v2 reward, which is added to other rewards
        transition_penalty_weight: weight for transition penalty, which is subtracted from other rewards
        similarity_penalty_weight: weight for similarity penalty, which is subtracted from other rewards
        sample_correctness_weight: weight for sample correctness reward, which is added to other rewards
        sample_count_penalty_weight: weight for sample count penalty, which is subtracted from other rewards
        reward_min: minimum reward value (lower bound for clipping)
        reward_max: maximum reward value (upper bound for clipping)
        extra_info: Additional info dict
    """
    if extra_info is None:
        extra_info = {}
    elif not isinstance(extra_info, dict):
        extra_info = json.loads(extra_info)

    assert isinstance(extra_info, dict)
    assert 'task_source' in extra_info, "extra_info must contain 'task_source' key"

    # evaluate model response
    ds = Dataset.from_list([{ **extra_info, 'model_responses__verl': [solution_str], 'answer': ground_truth, 'task_source': data_source }])
    answer_tag_pattern = TagPattern('<answer>', '</answer>', 'answer tag', 1.0)
    extractor = AnswerExtractor([answer_tag_pattern], [answer_tag_pattern])
    result = evaluate_dataset(ds, ['model_responses__verl'], verbose=False, custom_extractor=extractor)

    # all intermediate answers that are in <answer> tags (including the final answer so you may want to ignore the last one.)
    # internal_answers = result['model_responses__verl__internal_answers__eval_extracted_answers']

    # final answer
    assert len(result['model_responses__verl__eval_is_correct']) == 1 # one dataset row
    assert len(result['model_responses__verl__eval_is_correct'][0]) == 1 # one solution_str
    is_correct = result['model_responses__verl__eval_is_correct'][0][0]

    # intermediate answers (transition penalty)
    assert len(result['model_responses__verl__internal_answers__eval_is_correct']) == 1 # one dataset row
    transition_penalty = 0.0
    if result['model_responses__verl__internal_answers__eval_is_correct'][0] != None:
        assert len(result['model_responses__verl__internal_answers__eval_is_correct'][0]) == 1 # one solution_str
        transition_penalty = compute_transition_penalty(result['model_responses__verl__internal_answers__eval_is_correct'][0][0])

    # Compute formatting scores
    format_score = compute_format_score(solution_str)
    format_score_v2 = compute_format_score_v2(solution_str)
    
    # Compute similarity penalty
    similarity_penalty = compute_similarity_penalty(solution_str)
    
    # Compute sample correctness reward
    sample_correctness_reward = 0.0
    if result['model_responses__verl__internal_answers__eval_is_correct'][0] != None:
        assert len(result['model_responses__verl__internal_answers__eval_is_correct'][0]) == 1 # one solution_str
        sample_correctness_reward = compute_sample_correctness_reward(solution_str, result['model_responses__verl__internal_answers__eval_is_correct'][0][0])
    
    # Compute sample count penalty
    sample_count_penalty = compute_sample_count_penalty(solution_str)

    # Combine rewards
    final_reward = is_correct \
        + format_score * format_score_weight \
        + format_score_v2 * format_score_v2_weight \
        - transition_penalty * transition_penalty_weight \
        - similarity_penalty * similarity_penalty_weight \
        + sample_correctness_reward * sample_correctness_weight \
        - sample_count_penalty * sample_count_penalty_weight

    # Clip reward to bounds
    final_reward = max(reward_min, min(reward_max, final_reward))

    # Return both the final reward and the individual components for logging
    return {
        "score": final_reward, # the actual reward used for training
        "final_is_correct": float(is_correct), # for logging
        "format_score": format_score, # for logging
        "format_score_v2": format_score_v2, # for logging
        "transition_penalty": transition_penalty, # for logging
        "similarity_penalty": similarity_penalty, # for logging
        "sample_correctness_reward": sample_correctness_reward, # for logging
        "sample_count_penalty": sample_count_penalty, # for logging
    }

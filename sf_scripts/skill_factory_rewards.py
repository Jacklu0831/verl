from skill_factory.data.sources.clean_evaluation_helpers import evaluate_dataset
from skill_factory.data.sources.answer_extraction import TagPattern, AnswerExtractor

from datasets import Dataset
import json

def compute_score(data_source, solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, extra_info=None):
    """
    """

    if extra_info is not None and not isinstance(extra_info, dict):
        extra_info = json.loads(extra_info)

    assert 'task_source' in extra_info, "extra_info must contain 'task_source' key"

    ds = Dataset.from_list([
        { **extra_info, 'model_responses__verl': [solution_str], 'answer': ground_truth, 'task_source': data_source }
    ])

    answer_tag_pattern = TagPattern('<answer>', '</answer>', 'answer tag', 1.0)
    extractor = AnswerExtractor(
        [answer_tag_pattern],
        [answer_tag_pattern]
    )
    result = evaluate_dataset(ds, ['model_responses__verl'], verbose=False, custom_extractor=extractor)

    # all intermediate answers that are in <answer> tags (including the final answer so you may want to ignore the last one.)
    # internal_answers = result['model_responses__verl__internal_answers__eval_extracted_answers']

    is_correct = result['model_responses__verl__eval_is_correct'][0][0]

    if is_correct:
        return score
    else:
        return format_score
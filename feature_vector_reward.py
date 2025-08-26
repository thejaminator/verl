import asyncio
from typing import Any

from slist import Slist

from detection_eval.caller import (
    Caller,
    ChatHistory,
    InferenceConfig,
    load_multi_caller,
    read_jsonl_file_into_basemodel,
)
from detection_eval.detection_basemodels import SAEV2, SAEVerlData, SAEVerlDataTypedDict
from detection_eval.detection_basemodels import SAEActivationsV2 as SAEActivationsV2
from eval_detection import evaluate_sentence_matching
from eval_detection_v2 import (
    DetectionResult,
    SAETrainTestWithExplanation,
    _sentence_text_v2,
    create_detection_batch,
    evaluate_sentence_matching,
)


def has_opening_explanation_tag(solution_str: str) -> bool:
    return "<explanation>" in solution_str


def has_closing_explanation_tag(solution_str: str) -> bool:
    return "</explanation>" in solution_str


def parse_explanation(solution_str: str) -> str | None:
    if has_opening_explanation_tag(solution_str) and has_closing_explanation_tag(solution_str):
        return solution_str.split("<explanation>")[1].split("</explanation>")[0]
    return None


def verl_sample_sentences(
    sae: SAEVerlData,
    explanation: str,
    test_target_activating_sentences: Slist[int],
    train_activating_sentences: int,
    train_hard_negative_sentences: int,
    train_hard_negative_saes: int,
    test_hard_negative_sentences: int,
    test_hard_negative_saes: int,
) -> SAETrainTestWithExplanation | None:
    """Construct train/test activations and hard negatives from SAEVerlData.

    This mirrors the splitting logic used elsewhere, but consumes already-
    materialized positive and negative examples from `SAEVerlData`.
    """
    if len(sae.activations.sentences) == 0:
        return None

    # Extract positive examples for training and testing
    all_positive_sentences = Slist(sae.activations.sentences)

    # Determine how many test sentences to take (sample deterministically by SAE ID)
    sampled_test_sentences: int = test_target_activating_sentences.sample(n=1, seed=str(sae.sae_id))[0]

    # Ensure we have enough positives overall
    needed_sentences = sampled_test_sentences + train_activating_sentences
    if len(all_positive_sentences) < needed_sentences:
        print(
            f"WARNING: Not enough positive sentences to split into train/test for SAE {sae.sae_id}: "
            f"{len(all_positive_sentences)}, needed {needed_sentences}"
        )
        return None

    # Shuffle deterministically and filter out empty sentences
    shuffled_positive = all_positive_sentences.shuffle(str(sae.sae_id)).filter(lambda x: _sentence_text_v2(x) != "")

    train_positive_sentences = shuffled_positive[:train_activating_sentences]
    test_positive_sentences = shuffled_positive[
        train_activating_sentences : train_activating_sentences + sampled_test_sentences
    ]

    train_activations = SAEActivationsV2(sae_id=sae.sae_id, sentences=train_positive_sentences)
    test_activations = SAEActivationsV2(sae_id=sae.sae_id, sentences=test_positive_sentences)

    # Build hard negatives for train/test
    train_hard_negatives: list[SAEActivationsV2] = []
    test_hard_negatives: list[SAEActivationsV2] = []

    # Filter negative SAEs with sufficient sentences
    valid_train_hard_negs = [neg for neg in sae.hard_negatives if len(neg.sentences) >= train_hard_negative_sentences]
    valid_test_hard_negs = [neg for neg in sae.hard_negatives if len(neg.sentences) >= test_hard_negative_sentences]

    if train_hard_negative_saes > 0 and len(valid_train_hard_negs) < train_hard_negative_saes:
        print(
            f"WARNING: Not enough valid train hard negative SAEs for SAE {sae.sae_id}: "
            f"{len(valid_train_hard_negs)} available, requires {train_hard_negative_saes}"
        )
        return None

    if test_hard_negative_saes > 0 and len(valid_test_hard_negs) < test_hard_negative_saes:
        print(
            f"WARNING: Not enough valid test hard negative SAEs for SAE {sae.sae_id}: "
            f"{len(valid_test_hard_negs)} available, requires {test_hard_negative_saes}"
        )
        return None

    # Sample train hard-negative SAEs deterministically, then pick sentences
    if train_hard_negative_saes > 0:
        selected_train_hard_negs = Slist(valid_train_hard_negs).sample(
            n=train_hard_negative_saes, seed=f"{sae.sae_id}_train"
        )
        for neg_sae in selected_train_hard_negs:
            shuffled_neg_sentences = Slist(neg_sae.sentences).shuffle(str(neg_sae.sae_id))
            filtered_neg_sentences = shuffled_neg_sentences.filter(lambda x: _sentence_text_v2(x) != "")
            train_neg_sentences = filtered_neg_sentences[:train_hard_negative_sentences]
            if train_neg_sentences:
                train_hard_negatives.append(SAEActivationsV2(sae_id=neg_sae.sae_id, sentences=train_neg_sentences))

    # Sample test hard-negative SAEs deterministically, then pick sentences
    if test_hard_negative_saes > 0:
        selected_test_hard_negs = Slist(valid_test_hard_negs).sample(
            n=test_hard_negative_saes, seed=f"{sae.sae_id}_test"
        )
        for neg_sae in selected_test_hard_negs:
            shuffled_neg_sentences = Slist(neg_sae.sentences).shuffle(str(neg_sae.sae_id))
            filtered_neg_sentences = shuffled_neg_sentences.filter(lambda x: _sentence_text_v2(x) != "")
            test_neg_sentences = filtered_neg_sentences[:test_hard_negative_sentences]
            if test_neg_sentences:
                test_hard_negatives.append(SAEActivationsV2(sae_id=neg_sae.sae_id, sentences=test_neg_sentences))

    return SAETrainTestWithExplanation(
        sae_id=sae.sae_id,
        train_activations=train_activations,
        test_activations=test_activations,
        train_hard_negatives=train_hard_negatives,
        test_hard_negatives=test_hard_negatives,
        explanation=ChatHistory().add_assistant(content=explanation),
        explainer_model="verl",
    )


async def run_detection_with_verl_format(
    sae: SAETrainTestWithExplanation, caller: Caller, detection_config: InferenceConfig
) -> DetectionResult | None:
    """
    Run detection for SAEVerlData.
    Note: since we use feature steering, I didn't implement train vs test sentences
    IF we want to compare directly, should implement same split in future.
    """
    # turn into class MixedSentencesBatch(BaseModel):
    mixed_sentences_batch = create_detection_batch(sae)

    return await evaluate_sentence_matching(
        batch=mixed_sentences_batch, caller=caller, explainer_model="verl", detection_config=detection_config
    )


async def compute_score_single(explanation: str, sae: SAEVerlData, caller: Caller) -> float:
    """
    Custom reward function for math problems using proper verl interface.

    Args:
        data_source: The input data/prompt (not used in this implementation)
        solution_str: The model's generated response/solution
        ground_truth: The expected answer
        extra_info: Dict containing batch information, expects 'all_lengths' key with list of lengths

    Returns:
        Float reward score for this single example
    """
    explanation_parsed = parse_explanation(explanation)
    if explanation_parsed is None:
        print(f"WARNING: No parsed explanation for {sae.sae_id}. Explanation: {explanation}")
        return 0.0

    # turn into SAETrainTestWithExplanation
    sae_train_test = verl_sample_sentences(
        sae=sae,
        explanation=explanation_parsed,
        test_target_activating_sentences=Slist([4, 5, 6, 7, 8]),
        train_activating_sentences=4,
        train_hard_negative_sentences=4,
        train_hard_negative_saes=4,
        test_hard_negative_sentences=4,
        test_hard_negative_saes=4,
    )
    if sae_train_test is None:
        print(f"WARNING: Not enough sentences for SAE train test for {sae.sae_id}")
        return 0.0

    # run detection
    detection_result = await run_detection_with_verl_format(
        sae_train_test, caller, InferenceConfig(model="gpt-4o-mini")
    )
    if detection_result is None:
        return 0.0

    total_reward = detection_result.f1_score  ## 0 to 100

    return total_reward


caller = load_multi_caller(cache_path="cache/detection_eval")


def _compute_score(solution_str: list[str], parsed_sae: list[SAEVerlData]) -> list[float]:
    assert len(solution_str) == len(parsed_sae)
    explanation_sae = Slist(solution_str).zip(parsed_sae)

    # Run the async function in a synchronous context
    loop = asyncio.get_event_loop()
    print(f"Computing f1 rewards for {len(explanation_sae)} examples")
    result = loop.run_until_complete(
        explanation_sae.par_map_async(lambda pair: compute_score_single(pair[0], pair[1], caller=caller), tqdm=True)
    )
    return result


def compute_score(
    data_source: list[str], solution_str: list[str], ground_truth: list[str | None], extra_info: list[dict[str, Any]]
) -> list[float]:
    sae: list[SAEVerlDataTypedDict] = [i["sae"] for i in extra_info]
    parsed_sae: list[SAEVerlData] = [SAEVerlData.from_typed_dict(i) for i in sae]
    return _compute_score(solution_str, parsed_sae)


if __name__ == "__main__":
    saes = (
        read_jsonl_file_into_basemodel(path="data/hard_negatives_0_to_200.jsonl", basemodel=SAEV2)
        .map(lambda x: SAEVerlData.from_sae(x, feature_vector=[0.0] * 100, position_id=0))
        .take(1)
    )
    solution_str = ["<explanation>specifications and features of performance vehicles</explanation>"]
    print(_compute_score(solution_str, saes))

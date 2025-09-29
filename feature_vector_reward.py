import asyncio
import math
import threading
from typing import Any

from slist import Slist

from detection_eval.caller import (
    Caller,
    ChatHistory,
    InferenceConfig,
    load_pooled_openai_caller,
    read_jsonl_file_into_basemodel,
)
from detection_eval.detection_basemodels import SAEV2, SAEVerlData, SAEVerlDataTypedDict
from detection_eval.detection_basemodels import SAEActivationsV2 as SAEActivationsV2
from eval_detection_v2_judge_variance import (
    DetectionResult,
    SAETrainTestWithExplanation,
    _sentence_text_v2,
    create_detection_batch,
    evaluate_sentence_matching_repeated,
)

# Background event loop management (single-thread, no locking needed if no concurrency)
_BG_LOOP: asyncio.AbstractEventLoop | None = None
_BG_THREAD: threading.Thread | None = None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _BG_LOOP, _BG_THREAD
    if _BG_LOOP is not None and _BG_THREAD is not None and _BG_THREAD.is_alive():
        return _BG_LOOP

    loop = asyncio.new_event_loop()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_runner, name="verl-bg-loop", daemon=True)
    thread.start()

    _BG_LOOP = loop
    _BG_THREAD = thread
    return loop


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

REPEATS_JUDGE = 16

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

    return await evaluate_sentence_matching_repeated(
        batch=mixed_sentences_batch, caller=caller, explainer_model="verl", detection_config=detection_config, repeats=REPEATS_JUDGE
    )


detection_config = InferenceConfig(
    model="gpt-5-mini-2025-08-07",
    max_completion_tokens=10_000,
    reasoning_effort="minimal",
    temperature=1.0,
)


async def compute_score_single(explanation: str, sae: SAEVerlData, caller: Caller) -> DetectionResult | None:
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
    # explanation_parsed = parse_explanation(explanation)
    # if explanation_parsed is None:
    #     print(f"WARNING: No parsed explanation for {sae.sae_id}. Explanation: {explanation}")
    #     return None

    # turn into SAETrainTestWithExplanation
    sae_train_test = verl_sample_sentences(
        sae=sae,
        explanation=explanation,
        test_target_activating_sentences=Slist([4, 5, 6, 7, 8]),
        train_activating_sentences=1,
        train_hard_negative_sentences=1,
        train_hard_negative_saes=1,
        # Note: The "train" ones don't matter if just using feature vector,
        # since they don't appear in the prompt.
        test_hard_negative_saes=4,
        test_hard_negative_sentences=8,
    )
    if sae_train_test is None:
        print(f"WARNING: Not enough sentences for SAE train test for {sae.sae_id}")
        return None

    # run detection
    detection_result: DetectionResult | None = await run_detection_with_verl_format(
        sae_train_test, caller, detection_config
    )
    return detection_result


REWARD_CALLER = load_pooled_openai_caller(cache_path="/tmp/detection_eval")


async def compute_scores(
    explanation: list[str], sae: list[SAEVerlData], caller: Caller
) -> list[DetectionResult | None]:
    assert len(explanation) == len(sae)
    try:
        async with caller:
            # makes sure caller is flushed to write to cache
            result = (
                await Slist(explanation)
                .zip(sae)
                .par_map_async(lambda pair: compute_score_single(pair[0], pair[1], caller))
            )
        return result
    except Exception as e:
        print(f"ERROR in compute_scores: {e}")
        import traceback

        traceback.print_exc()
        # Return None for each input as fallback
        raise e


def fire_and_forget_compute_score(explanation: list[str], sae: list[SAEVerlData]) -> None:
    # For use during rollouts so that we can pre-compute reward scores without waiting for all rollouts to finish.
    # If there's a running loop, schedule on it; otherwise spin up a background loop.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(compute_scores(explanation, sae, REWARD_CALLER))
    except RuntimeError:
        # Schedule on a single background loop/thread and return immediately
        background_loop = _ensure_background_loop()
        asyncio.run_coroutine_threadsafe(compute_scores(explanation, sae, REWARD_CALLER), background_loop)


def bin_score(score: float) -> float:
    score = math.floor(score * 10) / 10
    # if perfect score, make it same as 0.9 (Perfect isn't that much better than 0.9, accounting for noise.)
    if score == 1.0:
        score = 0.90
    return score


def _compute_score(solution_str: list[str], parsed_sae: list[SAEVerlData]) -> list[float]:
    assert len(solution_str) == len(parsed_sae)
    print_strings = min(len(solution_str), 4)
    # for i in range(print_strings):
    #     print(f"String {i}: {solution_str[i]}")

    #
    explanation_sae = Slist(solution_str).zip(parsed_sae)

    # Run the async function in a synchronous context
    loop = asyncio.get_event_loop()
    # print(f"Computing f1 rewards for {len(explanation_sae)} examples")
    result = loop.run_until_complete(
        explanation_sae.par_map_async(
            lambda pair: compute_score_single(pair[0], pair[1], caller=REWARD_CALLER), tqdm=True
        )
    )
    first_result = result.filter(lambda x: x is not None).first_option
    if first_result is not None:
        # pritn the log
        last_message = first_result.evaluation_history.messages[-1]
        print(f"Eval log: {last_message.content}")

    to_rewards = result.map(lambda x: x.f1_score if x is not None else 0.0)
    # discretize into 0.2
    # if bin_scores:
    # Discretize scores into bins of 0.2
    # experiment : disable binning
    to_rewards = to_rewards.map(bin_score)

    return to_rewards


def compute_score(
    data_source: list[str], solution_str: list[str], sae: list[SAEVerlDataTypedDict], extra_info: list[dict[str, Any]]
) -> list[float]:
    parsed_sae: list[SAEVerlData] = [SAEVerlData.from_typed_dict(i) for i in sae]
    return _compute_score(solution_str, parsed_sae)


if __name__ == "__main__":
    saes = read_jsonl_file_into_basemodel(
        path="data/qwen_hard_negatives_20000_22000_layer_percent_25.jsonl", basemodel=SAEV2, limit=2
    ).map(lambda x: SAEVerlData.from_sae(x, feature_vector=[0.0] * 100, position_id=0))
    solution_str = [
        (
            "<explanation>Sentences sabouts NHRA-style drag racing events and related specialized "
            "drag-racing terminology — e.g., four-wide competitions, Funny Car/Top Fuel races or "
            "exhibitions, specific dragstrips/venues (zMAX Dragway, Texas Motorplex, MIR), event "
            "promotions/shoots (PINKS All Out), televised or exhibition race details, and "
            "fan/competition descriptions. These are event-focused mentions of drag-racing "
            "competitions and their jargon, distinguishing them from unrelated sports, "
            "entertainment, or general topics.</explanation>"
        ),
        (
            "<explanation>Short noun psshrases that cssharacterize a human with an evaluative or "
            "descriptive adjective (or adjective-like phrase) immediately before or after a head "
            "like 'man' or 'person' — e.g., 'a kind man,' 'a quiet man,' 'a humble man,' 'a smart "
            "person,' 'John Parish has been a busy man.' These are attributive character "
            "descriptions (including proverb-like patterns 'A smart man...') rather than neutral "
            "references to people or more complex syntactic uses (e.g., 'the man who is serving "
            "as...' or factual/organizational mentions), which do not activate the feature.</explanation>"
        ),
    ]
    print(_compute_score(solution_str, saes))

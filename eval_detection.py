import asyncio
from enum import unique
from typing import Sequence

import plotly.graph_objects as go
from openai import AsyncOpenAI, BaseModel
from pydantic import BaseModel
from slist import Group, Slist

from detection_eval.caller import (
    Caller,
    ChatHistory,
    ContentPolicyError,
    InferenceConfig,
    OpenAICaller,
    load_multi_caller,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from detection_eval.detection_basemodels import SAE, SAEActivations, SentenceInfo, TokenActivation


class ModelInfo(BaseModel):
    model: str
    display_name: str
    reasoning_effort: str | None = None


class SAEExplained(BaseModel):
    sae_id: int
    explanation: str
    positive_examples: list[str]
    negative_examples: list[str]
    f1: float


def read_sae_file(sae_file: str, limit: int | None = None, start_index: int = 0) -> Slist[SAE]:
    """Read SAEs from a JSONL file with optional start index and limit.

    Args:
        sae_file: Path to JSONL file.
        limit: Maximum number of SAEs to read. If None, read all from start_index.
        start_index: Number of initial lines to skip before reading.
    """
    if limit is None and start_index == 0:
        return read_jsonl_file_into_basemodel(sae_file, SAE)
    else:
        with open(sae_file) as f:
            output = Slist[SAE]()
            # Skip the first `start_index` lines
            for _ in range(start_index):
                try:
                    next(f)
                except StopIteration:
                    return output
            # Read up to `limit` items (or rest of file if limit is None)
            for line in f:
                sae = SAE.model_validate_json(line)
                output.append(sae)
                if limit is not None and len(output) >= limit:
                    break
            return output


def sentence_to_prompt_with_vector(sentence: SentenceInfo) -> str:
    """
    Convert a SentenceInfo object to a prompt.
    """
    try:
        max_activation_token: TokenActivation | None = Slist(sentence.tokens).max_by(lambda x: x.activation)
    except Exception as e:
        print(f"Error converting sentence to prompt: {e}")
        print(sentence)
        raise e
    assert max_activation_token is not None, f"No max activation token for sentence: {sentence}"
    activation_vector = sentence.as_activation_vector()
    return f"""<full_sentence>
{sentence.as_str}
</full_sentence>
<max_activation>
{sentence.max_activation}
</max_activation>
<max_activation_token>
{max_activation_token.as_str}
</max_activation_token>
<activation_vector>
{activation_vector}
</activation_vector>"""


def sentence_to_prompt_text_only(sentence: SentenceInfo) -> str:
    """
    Convert a SentenceInfo object to a prompt.
    """
    try:
        max_activation_token: TokenActivation | None = Slist(sentence.tokens).max_by(lambda x: x.activation)
    except Exception as e:
        print(f"Error converting sentence to prompt: {e}")
        print(sentence)
        raise e
    return f"""<full_sentence>
{sentence.as_str}
</full_sentence>"""


class SAETrainTest(BaseModel):
    sae_id: int
    feature_vector: Sequence[float]
    train_activations: SAEActivations
    test_activations: SAEActivations
    # Sentences that do not activate for the given sae_id. But come from a similar SAE
    # Here the sae_id correspond to different similar SAEs.
    # The activations are the activations w.r.t this SAE. And should be low.
    train_hard_negatives: list[SAEActivations]
    test_hard_negatives: list[SAEActivations]

    @staticmethod
    def from_sae(
        sae: SAE,
        target_feature_test_sentences: int,
        target_feature_train_sentences: int,
        train_hard_negative_saes: int,
        train_hard_negative_sentences: int,
        test_hard_negative_saes: int,
        test_hard_negative_sentences: int,
    ) -> "SAETrainTest | None":
        # Split main activations into train/test
        needed_sentences = target_feature_test_sentences + target_feature_train_sentences
        if len(sae.activations.sentences) < needed_sentences:
            print(
                f"WARNING: Not enough sentences to split into train and test: {len(sae.activations.sentences)}, needed {needed_sentences}"
            )
            return None
        # assert len(sae.activations.sentences) >= needed_sentences, (
        #     f"Not enough sentences to split into train and test: {len(sae.activations.sentences)}, needed {needed_sentences}"
        # )
        # TODO: Fix upstream code to filtered for non empty sentences.
        shuffled_sentences = Slist(sae.activations.sentences).shuffle(str(sae.sae_id)).filter(lambda x: x.as_str != "")

        train_sentences = shuffled_sentences[:target_feature_train_sentences]
        test_sentences = shuffled_sentences[
            target_feature_train_sentences : target_feature_train_sentences + target_feature_test_sentences
        ]

        train_activations = SAEActivations(sae_id=sae.sae_id, sentences=train_sentences)
        test_activations = SAEActivations(sae_id=sae.sae_id, sentences=test_sentences)

        # Split hard negatives into train/test
        train_hard_negatives: list[SAEActivations] = []
        test_hard_negatives: list[SAEActivations] = []

        # Filter hard negatives that have enough sentences for training
        valid_train_hard_negatives = [
            hard_neg for hard_neg in sae.hard_negatives if len(hard_neg.sentences) >= train_hard_negative_sentences
        ]

        # Filter hard negatives that have enough sentences for testing
        valid_test_hard_negatives = [
            hard_neg for hard_neg in sae.hard_negatives if len(hard_neg.sentences) >= test_hard_negative_sentences
        ]

        if len(valid_train_hard_negatives) < train_hard_negative_saes:
            print(
                f"WARNING: Not enough valid hard negative SAEs for training: {len(valid_train_hard_negatives)} available, {train_hard_negative_saes} required (each needing {train_hard_negative_sentences} sentences)"
            )
            return None

        if len(valid_test_hard_negatives) < test_hard_negative_saes:
            print(
                f"WARNING: Not enough valid hard negative SAEs for testing: {len(valid_test_hard_negatives)} available, {test_hard_negative_saes} required (each needing {test_hard_negative_sentences} sentences)"
            )
            return None

        # Sample train hard negatives
        selected_train_hard_negatives = Slist(valid_train_hard_negatives).sample(
            n=train_hard_negative_saes, seed=f"{sae.sae_id}_train"
        )

        for hard_negative_sae in selected_train_hard_negatives:
            shuffled_hard_neg_sentences = Slist(hard_negative_sae.sentences).shuffle(str(hard_negative_sae.sae_id))
            # For some reason, the hard negative sentences are sometimes empty.
            filtered_hard_neg_sentences: Slist[SentenceInfo] = shuffled_hard_neg_sentences.filter(
                lambda x: x.as_str != ""
            )
            train_hard_neg_sentences = filtered_hard_neg_sentences[:train_hard_negative_sentences]

            if train_hard_neg_sentences:
                train_hard_negatives.append(
                    SAEActivations(sae_id=hard_negative_sae.sae_id, sentences=train_hard_neg_sentences)
                )

        # Sample test hard negatives (can be different from training ones)
        selected_test_hard_negatives = Slist(valid_test_hard_negatives).sample(
            n=test_hard_negative_saes, seed=f"{sae.sae_id}_test"
        )

        for hard_negative_sae in selected_test_hard_negatives:
            shuffled_hard_neg_sentences = Slist(hard_negative_sae.sentences).shuffle(str(hard_negative_sae.sae_id))
            test_hard_neg_sentences = shuffled_hard_neg_sentences[:test_hard_negative_sentences]

            if test_hard_neg_sentences:
                test_hard_negatives.append(
                    SAEActivations(sae_id=hard_negative_sae.sae_id, sentences=test_hard_neg_sentences)
                )

        total_train_hard_negatives = len(train_hard_negatives) * train_hard_negative_sentences
        total_test_hard_negatives = len(test_hard_negatives) * test_hard_negative_sentences
        print(
            f"SAE {sae.sae_id} has {total_train_hard_negatives} train hard negatives and {total_test_hard_negatives} test hard negatives"
        )

        return SAETrainTest(
            sae_id=sae.sae_id,
            feature_vector=sae.feature_vector,
            train_activations=train_activations,
            test_activations=test_activations,
            train_hard_negatives=train_hard_negatives,
            test_hard_negatives=test_hard_negatives,
        )


def format_sae_prompt_for_explanation(activation: SAETrainTest) -> str:
    """
    Convert SAETrainTest to a prompt including positive examples and hard negatives.
    """
    prompt = ""

    # Add positive activating examples
    for idx, sentence in enumerate(activation.train_activations.sentences):
        prompt += f"<positive_example_{idx}>\n"
        prompt += sentence_to_prompt_with_vector(sentence)
        prompt += f"</positive_example_{idx}>\n"

    # Add hard negatives (sentences that do NOT activate this feature)
    negative_idx = 0
    for hard_neg_sae in activation.train_hard_negatives:
        for sentence in hard_neg_sae.sentences:
            prompt += f"<negative_example_{negative_idx}>\n"
            prompt += sentence_to_prompt_with_vector(sentence)
            prompt += f"</negative_example_{negative_idx}>\n"
            negative_idx += 1

    prompt += """\nThe above examples show sentences that activate a Sparse Auto Encoder (SAE) feature (positive examples) and sentences that do NOT activate this feature (negative examples). What concept do you think this SAE feature explains?

The positive examples are sentences that strongly activate the feature, while the negative examples are sentences from similar features that do NOT activate this particular feature. Use both the positive and negative examples to understand what makes this feature unique.

When writing the explanation, explain straightaway, don't say 'This feature detects...' or 'This feature explains...' or anything like that. 
1. Just try to explain the concept that the positive examples have in common that distinguishes them from the negative examples.
2. Your explanation will be passed to another human /model that will try to understand the explanation you've written. So, be clear in your explanation.
3. Your explanation should be precise. It should be broad enough to cover the meaning of the positive examples. But at the same time, it should be precise enough to distinguish from the negative examples.

Please write your final answer of what this SAE feature explains in the following format:
<explanation>
...
</explanation>"""
    return prompt


class SAETrainTestWithExplanation(BaseModel):
    sae_id: int
    feature_vector: Sequence[float]
    train_activations: SAEActivations
    test_activations: SAEActivations
    train_hard_negatives: list[SAEActivations]
    test_hard_negatives: list[SAEActivations]
    explanation: ChatHistory
    explainer_model: str

    @property
    def explanation_text(self) -> str:
        return extract_explanation_text(self.explanation.messages[-1].content)

    def replace_explanation(self, explanation: ChatHistory, explainer_model: str) -> "SAETrainTestWithExplanation":
        new = self.model_copy()
        new.explanation = explanation
        new.explainer_model = explainer_model
        return new


class MixedSentencesBatch(BaseModel):
    """Contains mixed sentences from multiple SAEs for evaluation."""

    target_sae_id: int
    explanation_history: ChatHistory
    target_explanation: str
    positive_examples: list[SentenceInfo]  # Sentences that should activate the feature
    negative_examples: list[SentenceInfo]  # Sentences that should NOT activate the feature
    shuffled_sentences: list[SentenceInfo]  # All sentences shuffled for evaluation
    target_indices: set[int]  # Indices in shuffled_sentences that correspond to positive examples


class DetectionResult(BaseModel):
    """Results of precision/recall evaluation."""

    target_sae_id: int
    predicted_indices: set[int]
    true_indices: set[int]
    precision: float
    recall: float
    f1_score: float
    explanation_used: str
    evaluation_response: str
    explanation_history: ChatHistory
    evaluation_history: ChatHistory
    explainer_model: str
    positive_examples: list[str]
    negative_examples: list[str]

    def to_sae_explained(self) -> SAEExplained:
        # just 5 each?
        return SAEExplained(
            sae_id=self.target_sae_id,
            explanation=self.explanation_used,
            positive_examples=self.positive_examples[:5],
            negative_examples=self.negative_examples[:5],
            f1=self.f1_score,
        )


async def call_model_for_sae_explanation(
    activation: SAETrainTest, caller: Caller, model_info: ModelInfo, best_of_n: int | None
) -> Slist[SAETrainTestWithExplanation]:  # Best of n
    """
    Call the specified model to get an explanation for the SAE feature.
    """
    prompt = format_sae_prompt_for_explanation(activation)

    config = InferenceConfig(
        model=model_info.model,
        max_completion_tokens=10_000 if "claude" not in model_info.model else None,
        max_tokens=10_000 if "claude" in model_info.model else None,
        reasoning_effort=model_info.reasoning_effort,
        temperature=1.0,
        n=1 if best_of_n is None else best_of_n,
    )

    chat_history = ChatHistory().add_user(content=prompt)
    # prefill assistant sside for gemma cos gemma is dumb
    # if "gemma" in model_info.model:
    #     chat_history = chat_history.add_assistant(content="<explanation>")
    response = await caller.call(chat_history, config)
    if best_of_n is None:
        return Slist(
            [
                SAETrainTestWithExplanation(
                    sae_id=activation.sae_id,
                    feature_vector=activation.feature_vector,
                    train_activations=activation.train_activations,
                    test_activations=activation.test_activations,
                    train_hard_negatives=activation.train_hard_negatives,
                    test_hard_negatives=activation.test_hard_negatives,
                    explanation=chat_history.add_assistant(content=response.first_response.strip()),
                    explainer_model=model_info.model,
                )
            ]
        )
    else:
        return Slist(response.responses).map(
            lambda x: SAETrainTestWithExplanation(
                sae_id=activation.sae_id,
                feature_vector=activation.feature_vector,
                train_activations=activation.train_activations,
                test_activations=activation.test_activations,
                train_hard_negatives=activation.train_hard_negatives,
                test_hard_negatives=activation.test_hard_negatives,
                explanation=chat_history.add_assistant(content=x.strip()),
                explainer_model=model_info.model,
            )
        )


def extract_explanation_text(explanation_response: str) -> str:
    """Extract the explanation text from between XML tags."""
    start_tag = "<explanation>"
    end_tag = "</explanation>"

    start_idx = explanation_response.find(start_tag)
    end_idx = explanation_response.find(end_tag)

    if start_idx != -1 and end_idx != -1:
        start_idx += len(start_tag)
        return explanation_response[start_idx:end_idx].strip()
    else:
        # Fallback: return the whole response if tags aren't found
        return explanation_response.strip().replace("<explanation>", "").replace("</explanation>", "")


def create_detection_batch(
    target_sae: SAETrainTestWithExplanation,
) -> MixedSentencesBatch:
    """
    Create a batch of mixed sentences: target SAE test sentences + hard negatives.

    Args:
        target_sae: The SAE whose explanation we're testing

    Returns:
        MixedSentencesBatch with positive and negative examples
    """
    # Use all test sentences from target SAE as positive examples
    positive_examples = target_sae.test_activations.sentences

    # Collect hard negative sentences from test_hard_negatives as negative examples
    negative_examples = []
    for hard_neg_sae in target_sae.test_hard_negatives:
        negative_examples.extend(hard_neg_sae.sentences)

    # Combine positive and negative examples and shuffle for evaluation
    all_sentences = positive_examples + negative_examples
    num_positive = len(positive_examples)

    # Create deterministic shuffle based on target SAE ID
    deterministic_seed = f"{target_sae.sae_id}"
    shuffled_indices = Slist(list(range(len(all_sentences)))).shuffle(deterministic_seed)
    shuffled_sentences = [all_sentences[i] for i in shuffled_indices]

    # Find where positive examples ended up after shuffling
    target_indices = set()
    for i, original_idx in enumerate(shuffled_indices):
        if original_idx < num_positive:  # Original positive example indices were 0 to num_positive-1
            target_indices.add(i)

    # Extract explanation text
    explanation_text = extract_explanation_text(target_sae.explanation.messages[-1].content)

    return MixedSentencesBatch(
        target_sae_id=target_sae.sae_id,
        explanation_history=target_sae.explanation,
        target_explanation=explanation_text,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        shuffled_sentences=shuffled_sentences,
        target_indices=target_indices,
    )


def create_evaluation_prompt(batch: MixedSentencesBatch) -> str:
    """Create prompt for GPT-5-mini to identify matching sentences."""
    num_sentences = len(batch.shuffled_sentences)

    prompt = f"""I will provide you with an SAE feature explanation and {num_sentences} sentences. Your task is to identify which sentence numbers correspond to the given explanation.

<explanation>
{batch.target_explanation}
</explanation>

<sentences>
"""

    for i, sentence in enumerate(batch.shuffled_sentences):
        sentence_str = sentence_to_prompt_text_only(sentence)
        prompt += f"<sentence_{i}>\n{sentence_str}\n</sentence_{i}>\n"

    prompt += """</sentences>

Please carefully analyze each sentence and determine which ones match the SAE feature explanation provided above.
You should expect to find between 4-8 matching sentences. Analyze each sentence independently and only include ones that match the explanation.
Provide your final answer as a JSON object with an "answer" field containing a list of sentence numbers that match the explanation."""

    return prompt


class AnswerSchema(BaseModel):
    answer: list[int]


async def evaluate_sentence_matching(
    batch: MixedSentencesBatch, caller: Caller, explainer_model: str, detection_config: InferenceConfig
) -> DetectionResult | None:
    """
    Use GPT-5-mini to identify which sentences match the explanation.

    Args:
        batch: Mixed sentences batch with target explanation
        caller: API caller
        explainer_model: The model that generated the explanation

    Returns:
        EvaluationResult with precision, recall, and F1 scores
    """
    prompt = create_evaluation_prompt(batch)
    chat_history = ChatHistory().add_user(content=prompt)

    # Keep evaluation model constant as requested (GPT-5-mini)
    config = detection_config
    try:
        response = await caller.call_with_schema(chat_history, config=config, schema=AnswerSchema)
    except ContentPolicyError as e:
        print(f"Content policy error: {e}")
        return None
    answer = response.answer

    predicted_indices = set(answer)

    # Calculate metrics
    true_positive = len(predicted_indices & batch.target_indices)

    precision = true_positive / len(predicted_indices) if len(predicted_indices) > 0 else 0.0
    recall = true_positive / len(batch.target_indices) if len(batch.target_indices) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    eval_log = chat_history.add_assistant(content=str(answer))

    # Add logging message with model response and correctness evaluation
    correctness_msg = f"Model predicted indices: {answer} | True indices: {list(batch.target_indices)} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1_score:.3f}"

    retrieved_sentences_with_emojis = []
    for i, sentence_idx in enumerate(answer):
        sentence = batch.shuffled_sentences[sentence_idx]
        emoji = "✅" if sentence_idx in batch.target_indices else "❌"
        sentence_text = sentence_to_prompt_text_only(sentence)
        retrieved_sentences_with_emojis.append(f"{emoji} {sentence_text}")

    retrieved_sentences_str = "\n".join(retrieved_sentences_with_emojis)
    given_explanation_str = batch.target_explanation

    eval_log = eval_log.add_assistant(content=correctness_msg).add_user(
        content="SAE ID: "
        + str(batch.target_sae_id)
        + "\n\n"
        + given_explanation_str
        + "\n\n"
        + retrieved_sentences_str
        + "\n\n"
        + correctness_msg
    )

    # Use the stored positive and negative examples from the batch
    positive_examples_text = Slist(batch.positive_examples).map(lambda x: x.as_str).shuffle(f"{batch.target_sae_id}")
    negative_examples_text = Slist(batch.negative_examples).map(lambda x: x.as_str).shuffle(f"{batch.target_sae_id}")

    return DetectionResult(
        target_sae_id=batch.target_sae_id,
        predicted_indices=predicted_indices,
        true_indices=batch.target_indices,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        explanation_used=batch.target_explanation,
        evaluation_response=str(answer),
        explanation_history=batch.explanation_history,
        evaluation_history=eval_log,
        explainer_model=explainer_model,
        positive_examples=positive_examples_text,
        negative_examples=negative_examples_text,
    )


async def generate_explanations_for_model(
    split_sae_activations: Slist[SAETrainTest],
    model_info: ModelInfo,
    caller: Caller,
    max_par: int,
    best_of_n: int | None,
) -> Slist[SAETrainTestWithExplanation]:
    """Generate explanations for all SAE activations using a specific model."""
    print(f"Generating explanations using {model_info.display_name}...")

    explanations: Slist[Slist[SAETrainTestWithExplanation]] = await split_sae_activations.par_map_async(
        lambda activation: call_model_for_sae_explanation(activation, caller, model_info, best_of_n),
        max_par=max_par,
        tqdm=True,
    )

    return explanations.flatten_list()


async def run_evaluation_for_explanations(
    explanations: Slist[SAETrainTestWithExplanation], caller: Caller, max_par: int, detection_config: InferenceConfig
) -> Slist[DetectionResult]:
    """Run precision/recall evaluation for a set of explanations using their hard negatives."""
    # Filter out explanations that don't have any test hard negatives
    explanations_with_hard_negatives = explanations.filter(lambda x: len(x.test_hard_negatives) > 0)

    if len(explanations_with_hard_negatives) == 0:
        print("Warning: No explanations have hard negatives for evaluation")
        return Slist([])

    print(
        f"Running precision/recall evaluation on {len(explanations_with_hard_negatives)} SAEs using their hard negatives..."
    )

    # Create evaluation batches
    detection_batches: list[tuple[MixedSentencesBatch, str]] = []
    for target_sae in explanations_with_hard_negatives:
        batch = create_detection_batch(target_sae=target_sae)
        detection_batches.append((batch, target_sae.explainer_model))

    print(f"Created {len(detection_batches)} evaluation batches")

    # Run evaluations
    _evaluation_results: Slist[DetectionResult | None] = await Slist(detection_batches).par_map_async(
        lambda batch_and_model: evaluate_sentence_matching(
            batch_and_model[0], caller, batch_and_model[1], detection_config
        ),
        max_par=max_par,
        tqdm=True,
    )
    evaluation_results = _evaluation_results.flatten_option()

    return evaluation_results


async def run_best_of_n_evaluation_for_explanations(
    explanations: Slist[SAETrainTestWithExplanation], caller: Caller, max_par: int, detection_config: InferenceConfig
) -> Slist[DetectionResult]:
    """
    Run evaluation for all explanations and return only the best F1 score per SAE ID.

    Args:
        explanations: All explanations (potentially multiple per SAE if best_of_n > 1)
        caller: API caller for evaluations
        max_par: Maximum parallel requests

    Returns:
        List of evaluation results with the best F1 score per SAE ID
    """
    # First, run evaluations for all explanations
    all_evaluation_results = await run_evaluation_for_explanations(explanations, caller, max_par, detection_config)

    if len(all_evaluation_results) == 0:
        return Slist([])

    # Group evaluation results by SAE ID
    grouped_by_sae_id: Slist[Group[int, Slist[DetectionResult]]] = all_evaluation_results.group_by(
        lambda x: x.target_sae_id
    )

    # For each SAE ID, pick the result with the highest F1 score
    best_results: Slist[DetectionResult] = grouped_by_sae_id.map(
        lambda group: group[1].max_by(lambda result: result.f1_score)
    ).flatten_option()

    print(f"Best-of-n selection: Reduced {len(all_evaluation_results)} results to {len(best_results)} (best per SAE)")

    return best_results


def plot_f1_scores_by_model(groupby_by_model: Slist[Group[str, Slist[DetectionResult]]], rename_map: dict[str, str] = {}) -> None:
    """Plot F1 scores by model using plotly."""
    # Extract model names and average F1 scores
    model_names = []
    avg_f1_scores = []

    for model_name, evaluation_results in groupby_by_model:
        if len(evaluation_results) == 0:
            continue  # Skip models with no results

        avg_f1 = evaluation_results.map(lambda x: x.f1_score).sum() / len(evaluation_results)
        model_names.append(rename_map.get(model_name, model_name))
        avg_f1_scores.append(avg_f1 * 100)  # Convert to percentage

    # Create the bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=model_names,
                y=avg_f1_scores,
                text=[f"{score:.1f}%" for score in avg_f1_scores],
                textposition="auto",
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title="F1 Score by Model on Detection Eval",
        xaxis_title="Model",
        yaxis_title="F1 Score (%)",
        font=dict(size=16),
        yaxis=dict(
            tickformat=".1f",
            ticksuffix="%",
            range=[0, 100],  # Set y-axis range from 0 to 100%
        ),
        showlegend=False,
        width=1000,
        height=600,
    )

    # Show the plot
    fig.show()


def plot_precision_vs_recall_by_model(groupby_by_model: Slist[Group[str, Slist[DetectionResult]]]) -> None:
    """Plot precision vs recall with models as dots using plotly."""
    # Extract model names and average precision/recall scores
    model_names = []
    avg_precisions = []
    avg_recalls = []

    for model_name, evaluation_results in groupby_by_model:
        if len(evaluation_results) == 0:
            continue  # Skip models with no results

        avg_precision = evaluation_results.map(lambda x: x.precision).sum() / len(evaluation_results)
        avg_recall = evaluation_results.map(lambda x: x.recall).sum() / len(evaluation_results)

        model_names.append(model_name)
        avg_precisions.append(avg_precision * 100)  # Convert to percentage
        avg_recalls.append(avg_recall * 100)  # Convert to percentage

    # Create the scatter plot
    fig = go.Figure(
        data=[
            go.Scatter(
                x=avg_recalls,
                y=avg_precisions,
                mode="markers+text",
                marker=dict(size=12, opacity=0.8),
                text=model_names,
                textposition="top center",
                textfont=dict(size=12),
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title="Precision vs Recall by Model",
        xaxis_title="Recall (%)",
        yaxis_title="Precision (%)",
        font=dict(size=16),
        xaxis=dict(
            tickformat=".1f",
            ticksuffix="%",
            range=[0, 100],  # Set x-axis range from 0 to 100%
        ),
        yaxis=dict(
            tickformat=".1f",
            ticksuffix="%",
            range=[0, 100],  # Set y-axis range from 0 to 100%
        ),
        showlegend=False,
        width=800,
        height=600,
    )

    # Show the plot
    fig.show()


class SAEExperimentConfig(BaseModel):
    test_target_activating_sentences: Slist[int] = Slist([4, 5, 6, 7, 8])
    train_activating_sentences: int
    train_hard_negative_sentences: int
    train_hard_negative_saes: int
    test_hard_negative_sentences: int
    test_hard_negative_saes: int
    saes_to_test: int
    sae_start_index: int = 0
    best_of_n: int | None = None

    def replace(
        self,
        target_feature_test_sentences: Slist[int] | None = None,
        train_activating_sentences: int | None = None,
        test_hard_negative_sentences: int | None = None,
        train_hard_negative_sentences: int | None = None,
        train_hard_negative_saes: int | None = None,
        test_hard_negative_saes: int | None = None,
        saes_to_test: int | None = None,
        best_of_n: int | None = None,
    ) -> "SAEExperimentConfig":
        copy = self.model_copy()
        if target_feature_test_sentences is not None:
            copy.test_target_activating_sentences = target_feature_test_sentences
        if train_activating_sentences is not None:
            copy.train_activating_sentences = train_activating_sentences
        if test_hard_negative_sentences is not None:
            copy.test_hard_negative_sentences = test_hard_negative_sentences
        if train_hard_negative_sentences is not None:
            copy.train_hard_negative_sentences = train_hard_negative_sentences
        if train_hard_negative_saes is not None:
            copy.train_hard_negative_saes = train_hard_negative_saes
        if test_hard_negative_saes is not None:
            copy.test_hard_negative_saes = test_hard_negative_saes
        if saes_to_test is not None:
            copy.saes_to_test = saes_to_test
        if best_of_n is not None:
            copy.best_of_n = best_of_n
        return copy


async def run_gemma_steering(
    sae: SAETrainTest,
    gemma_caller: OpenAICaller,
    lora_model: str,
    try_number: int = 1,
) -> SAETrainTestWithExplanation:
    """
    Run gemma steering for a single SAE.
    """
    history = ChatHistory.from_user("Can you explain to me what 'X' means? Format your final answer with <explanation>")
    response = await gemma_caller.call(
        messages=history,
        config=InferenceConfig(
            model=lora_model,
            max_tokens=2000,
            extra_body={
                "sae_index": sae.sae_id,  # the server will use this to get the SAE feature vector
            },
        ),
        try_number=try_number,
    )
    explanation = response.first_response.strip()
    # return the SAETrainTestWithExplanation
    return SAETrainTestWithExplanation(
        sae_id=sae.sae_id,
        explanation=history.add_assistant(explanation),
        explainer_model=lora_model,
        # note: technically we didn't use these "train" things but we'll just pass it on.
        train_hard_negatives=sae.train_hard_negatives,
        train_activations=sae.train_activations,
        feature_vector=sae.feature_vector,
        test_activations=sae.test_activations,
        test_hard_negatives=sae.test_hard_negatives,
    )


async def run_gemma_steering_best_of_n(
    sae: SAETrainTest, gemma_caller: OpenAICaller, lora_model: str, best_of_n: int
) -> Slist[SAETrainTestWithExplanation]:
    """
    Run gemma steering for a single SAE with best-of-n.
    """
    # Run gemma steering for each try number
    _explanations: Slist[SAETrainTestWithExplanation] = await Slist(range(best_of_n)).par_map_async(
        lambda try_number: run_gemma_steering(sae, gemma_caller, lora_model, try_number),
        max_par=best_of_n,  # already max_par in outer loop
    )
    return _explanations


def make_random_explanation(items: Slist[SAETrainTestWithExplanation], name: str) -> Slist[SAETrainTestWithExplanation]:
    # sort by for determinism
    uniques = items.sort_by(lambda x: (x.sae_id, x.explainer_model, x.explanation_text)).shuffle("42").distinct_by(lambda x: x.sae_id)
    # reshuffle for random explanation
    shuffled_explanations = uniques.shuffle("42")
    zipped = shuffled_explanations.zip(uniques)
    return zipped.map(lambda x: x[0].replace_explanation(x[1].explanation, name))





async def main(
    explainer_models: Slist[ModelInfo],
    use_gemma_steering: bool,
    add_random_explanations: bool,
    sae_file: str,
    config: SAEExperimentConfig,
    max_par: int = 10,
):
    """
    Main function to process SAE activations, get explanations, and run precision/recall evaluation.

    Args:
        sae_file: Path to the SAE hard negatives JSONL file
        explainer_models: List of models to use for generating explanations
        target_feature_test_sentences: Number of test sentences to use for each target SAE feature
        train_activating_sentences: Number of training sentences to use for each SAE
        test_hard_negative_sentences: Number of hard negative sentences to use for testing
        train_hard_negative_sentences: Number of hard negative sentences to use for training
        train_hard_negative_saes: Number of hard negative SAEs to sample from for training
        test_hard_negative_saes: Number of hard negative SAEs to sample from for testing
        saes_to_test: Number of SAE activations to process (default: 10)
        max_par: Maximum parallel requests (default: 10)
    """
    # Load SAE activations
    print("Loading SAE data...")
    target_saes_to_test = config.saes_to_test
    target_feature_test_sentences = config.test_target_activating_sentences
    train_activating_sentences = config.train_activating_sentences
    train_hard_negative_sentences = config.train_hard_negative_sentences
    train_hard_negative_saes = config.train_hard_negative_saes
    test_hard_negative_saes = config.test_hard_negative_saes
    test_hard_negative_sentences = config.test_hard_negative_sentences

    saes: Slist[SAE] = read_sae_file(sae_file, limit=target_saes_to_test, start_index=config.sae_start_index)
    print(f"Loaded {len(saes)} SAE entries starting at index {config.sae_start_index}")

    def create_sae_train_test(sae: SAE) -> SAETrainTest | None:
        # Sample deterministically from test_target_activating_sentences using SAE ID as seed
        sampled_test_sentences = target_feature_test_sentences.sample(n=1, seed=str(sae.sae_id))[0]
        return SAETrainTest.from_sae(
            sae,
            target_feature_test_sentences=sampled_test_sentences,
            target_feature_train_sentences=train_activating_sentences,
            train_hard_negative_saes=train_hard_negative_saes,
            train_hard_negative_sentences=train_hard_negative_sentences,
            test_hard_negative_saes=test_hard_negative_saes,
            test_hard_negative_sentences=test_hard_negative_sentences,
        )

    _split_sae_activations = saes.map(create_sae_train_test)

    # These are the "train" SAEs that we will use for explanation generation
    split_sae_activations: Slist[SAETrainTest] = _split_sae_activations.flatten_option()

    print(f"Loaded {len(split_sae_activations)} valid SAE entries")

    # Create caller
    caller = load_multi_caller(cache_path="cache/sae_explanations")
    # Custom caller for gemma

    # Generate explanations for each model
    async with caller:
        _explanations: Slist[Slist[SAETrainTestWithExplanation]] = await explainer_models.par_map_async(
            lambda model_info: generate_explanations_for_model(
                split_sae_activations, model_info, caller, max_par, config.best_of_n
            ),
            max_par=max_par,
        )
        non_gemma_explanations = _explanations.flatten_list()
        all_explanations: Slist[SAETrainTestWithExplanation] = non_gemma_explanations

    caller_for_eval = load_multi_caller(cache_path="cache/sae_evaluations")


    if use_gemma_steering:
        # Run gemma steering
        print("Running gemma steering")
        lora_model = "thejaminator/sae-introspection-lora"
        gemma_client = AsyncOpenAI(api_key="dummy api key", base_url="https://94nlcy6stx75yz-8000.proxy.runpod.net/v1")
        width = 131  # not cached by api call yet, so manually add to cache path
        gemma_caller = OpenAICaller(openai_client=gemma_client, cache_path=f"cache/steering_cache_{width}")
        best_of_n = config.best_of_n
        if best_of_n is not None:
            max_par_bon = max_par // best_of_n
            best_of_n_gemma_explanations: Slist[
                Slist[SAETrainTestWithExplanation]
            ] = await split_sae_activations.par_map_async(
                lambda sae: run_gemma_steering_best_of_n(sae, gemma_caller, lora_model, best_of_n),
                max_par=max_par_bon,
                tqdm=True,
            )
            _gemma_explanations = best_of_n_gemma_explanations.flatten_list()
        else:
            _gemma_explanations: Slist[SAETrainTestWithExplanation] = await split_sae_activations.par_map_async(
                lambda sae: run_gemma_steering(sae, gemma_caller, lora_model),
                max_par=max_par,
                tqdm=True,
            )
        all_explanations = all_explanations + _gemma_explanations
        explainer_models = explainer_models + [
            ModelInfo(model=lora_model, display_name="Light SFT Gemma<br>(Introspecting<br>feature vector)", reasoning_effort="medium")
        ]

    if add_random_explanations:
        name = "Random explanation"
        all_explanations = all_explanations + make_random_explanation(non_gemma_explanations, name)
        explainer_models = explainer_models + [
            ModelInfo(model=name, display_name=name, reasoning_effort="low")
        ]
        
    # Run evaluations for each model's explanations
    detection_config = InferenceConfig(
        model="gpt-5-mini-2025-08-07",
        # model="gpt-5-nano-2025-08-07",
        max_completion_tokens=10_000,
        reasoning_effort="low",  # seems good enough
        # reasoning_effort="medium",
        temperature=1.0,
    )

    async with caller_for_eval:
        # sort for deterministic results
        all_explanations_sorted = all_explanations.sort_by(lambda x: x.sae_id)

        # Choose evaluation function based on whether we're using best-of-n
        detection_func = (
            run_best_of_n_evaluation_for_explanations
            if config.best_of_n is not None
            else run_evaluation_for_explanations
        )

        detection_results_per_model: Slist[Slist[DetectionResult]] = await explainer_models.par_map_async(
            lambda model_info: detection_func(
                all_explanations_sorted.filter(lambda x: x.explainer_model == model_info.model),
                caller_for_eval,
                max_par,
                detection_config,
            ),
            max_par=len(explainer_models),
        )
        all_detection_results = detection_results_per_model.flatten_list()

    # Print summary statistics for each model
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS BY MODEL")
    print("=" * 50)

    groupby_by_model: Slist[Group[str, Slist[DetectionResult]]] = all_detection_results.group_by(
        lambda x: x.explainer_model
    )

    # Plot F1 scores by model
    rename_map = {m.model: m.display_name for m in explainer_models}
    plot_f1_scores_by_model(groupby_by_model, rename_map)

    # Plot precision vs recall by model
    plot_precision_vs_recall_by_model(groupby_by_model)

    for model_name, evaluation_results in groupby_by_model:
        if len(evaluation_results) == 0:
            print(f"\n{model_name}: No evaluation results (insufficient data)")
            continue

        avg_precision = evaluation_results.map(lambda x: x.precision).sum() / len(evaluation_results)
        avg_recall = evaluation_results.map(lambda x: x.recall).sum() / len(evaluation_results)
        avg_f1 = evaluation_results.map(lambda x: x.f1_score).sum() / len(evaluation_results)

        print(f"\n{model_name}:")
        print(f"  Evaluated {len(evaluation_results)} SAEs")
        print(f"  Average Precision: {avg_precision:.3f}")
        print(f"  Average Recall: {avg_recall:.3f}")
        print(f"  Average F1-Score: {avg_f1:.3f}")

        # Save detailed evaluation results
        safe_model_name = model_name.replace("/", "_").replace(":", "_")

        # Save the generated explanations
        explanation_output_file = f"sae_explanations_{safe_model_name}.jsonl"
        write_jsonl_file_from_basemodel(
            path=explanation_output_file, basemodels=evaluation_results.map(lambda x: x.explanation_history)
        )
        print(f"  Explanations saved to {explanation_output_file}")

        # eval_output_file = f"sae_evaluation_results_{safe_model_name}.jsonl"
        # write_jsonl_file_from_basemodel(path=eval_output_file, basemodels=evaluation_results)
        # print(f"  Detailed results saved to {eval_output_file}")

        history_output_file = f"sae_evaluation_history_{safe_model_name}.jsonl"
        write_jsonl_file_from_basemodel(
            path=history_output_file, basemodels=evaluation_results.map(lambda x: x.evaluation_history)
        )
        print(f"  History saved to {history_output_file}")

        sft_data = evaluation_results.map(lambda x: x.to_sae_explained()).filter(lambda x: x.f1 > 0.8)
        # Save the SAE explanations
        sae_explanations_output_file = f"sae_sfted_{safe_model_name}.jsonl"
        write_jsonl_file_from_basemodel(path=sae_explanations_output_file, basemodels=sft_data)
        print(f"  SAE explanations saved to {sae_explanations_output_file}")


if __name__ == "__main__":
    # Define explainer models to test
    explainer_models = Slist(
        [
            ModelInfo(model="gpt-5-mini-2025-08-07", display_name="GPT-5-mini<br>(extrospecting<br>activating sentences)", reasoning_effort="medium"),
            ModelInfo(model="meta-llama/llama-3-70b-instruct", display_name="Llama-3-70b<br>(extrospecting<br>activating sentences)"),
            # ModelInfo(model="gpt-5-mini-2025-08-07", display_name="GPT-5-mini", reasoning_effort="low"),
            # meta-llama/llama-3-70b-instruct
            # ModelInfo(model="gpt-4.1-2025-04-14", display_name="GPT-4.1"),
            # ModelInfo(model="gpt-4o-2024-08-06", display_name="GPT-4o"),
            # ModelInfo(model="claude-sonnet-4-20250514", display_name="Claude-3.5-Sonnet"),
            # # google/gemma-2-9b-it
            # ModelInfo(model="google/gemma-2-9b-it", display_name="Gemma-2-9b-it"),
            # # google/gemma-3-12b-it
            # ModelInfo(model="google/gemma-3-12b-it", display_name="Gemma-3-12b-it"),
            # qwen/qwen3-30b-a3b-instruct-2507
            # ModelInfo(model="qwen/qwen3-30b-a3b-instruct-2507", display_name="Qwen-3-30b-a3b-instruct-2507"),
        ]
    )

    # created with create_hard_negative_and_feature_vector.py
    sae_file = "data/10k_hard_negatives_results.jsonl"
    # For each target SAE, we have 10 hard negative related SAEs by cosine similarity.
    # Which to use for constructing explanations vs testing detection?
    saes_to_test = 100
    sae_start_index = 2_000  # not in train set for the trained model

    hard_negatives_config = SAEExperimentConfig(
        test_target_activating_sentences=Slist([4, 5, 6, 7, 8]),
        train_activating_sentences=16,
        train_hard_negative_sentences=2,  # provide 8 hard negatives for training
        train_hard_negative_saes=16,
        # Note: total 34 hard negative SAEs to sample from``
        test_hard_negative_saes=16,  # 16 * 6 = 96 hard negatives for testing
        test_hard_negative_sentences=6,
        saes_to_test=saes_to_test,
        best_of_n=None,  # Set to an integer (e.g., 3, 5) to enable best-of-n
        sae_start_index=sae_start_index,
    )

    no_train_hard_negatives_config = hard_negatives_config.replace(train_hard_negative_saes=0)
    eight_positive_examples_config = hard_negatives_config.replace(train_activating_sentences=8)
    four_positive_examples_config = hard_negatives_config.replace(train_activating_sentences=4)
    two_positive_examples = hard_negatives_config.replace(train_activating_sentences=2)
    best_of_8_config = hard_negatives_config.replace(best_of_n=8)

    # You can run the full pipeline with configurable parameters
    # Full pipeline:
    # asyncio.run(main(explainer_models=explainer_models, saes_to_test=50, max_par=10, train_activating_sentences=4))
    # 4, 8, 16, 25?
    # asyncio.run(main(explainer_models=explainer_models, saes_to_test=50, max_par=10, train_activating_sentences=8))
    asyncio.run(
        main(
            sae_file=sae_file,
            use_gemma_steering=True,
            explainer_models=explainer_models,
            add_random_explanations=True,
            config=hard_negatives_config,
            # config=best_of_8_config,
            # config=no_train_hard_negatives_config,
            # config=eight_positive_examples_config,
            # config=two_positive_examples,
            # config=four_positive_examples_config,
            max_par=80,
        )
    )

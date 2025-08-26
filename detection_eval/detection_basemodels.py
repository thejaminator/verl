from typing import Any, TypedDict

from openai import BaseModel
from pydantic import BaseModel
from slist import Slist


class TokenActivation(BaseModel):
    as_str: str
    activation: float
    token_id: int

    def to_prompt_str(self) -> str:
        return f"{self.as_str} ({self.activation:.2f})"


class SentenceInfo(BaseModel):
    max_activation: float
    tokens: list[TokenActivation]
    as_str: str

    def as_activation_vector(self) -> str:
        activation_vector = Slist(self.tokens).map(lambda x: x.to_prompt_str())
        return f"{activation_vector}"


class SAEActivations(BaseModel):
    sae_id: int
    sentences: list[SentenceInfo]


class SAE(BaseModel):
    sae_id: int
    # feature_vector: Sequence[float]
    activations: SAEActivations
    # Sentences that do not activate for the given sae_id. But come from a similar SAE
    # Here the sae_id correspond to different similar SAEs.
    # The activations are the activations w.r.t this SAE. And should be low.
    hard_negatives: list[SAEActivations]


#### Try out smaller V2 models that are smaller"""
class TokenActivationV2(BaseModel):
    s: str
    act: float
    pos: int  # position in tokens

    def to_prompt_str(self) -> str:
        activation = f"{self.act:.2f}" if self.act is not None else "0.00"
        return f"{self.s} ({activation})"


class SentenceInfoV2(BaseModel):
    max_act: float
    tokens: list[str]
    act_tokens: list[TokenActivationV2]

    # def as_activation_vector(self) -> str:
    #     activation_vector = Slist(self.tokens).map(lambda x: x.to_prompt_str())
    #     return f"{activation_vector}"

    # def as_str(self) -> str:
    #     return Slist(self.tokens).map(lambda x: x.s).mk_string("")


class SAEActivationsV2(BaseModel):
    sae_id: int
    sentences: list[SentenceInfoV2]


class SAEV2(BaseModel):
    sae_id: int
    activations: SAEActivationsV2
    # Sentences that do not activate for the given sae_id. But come from a similar SAE
    # Here the sae_id correspond to different similar SAEs.
    # The activations are the activations w.r.t this SAE. And should be low.
    hard_negatives: list[SAEActivationsV2]


class SAEVerlDataTypedDict(TypedDict):
    """Typed dict that gets passed around in verl"""

    sae_id: int
    feature_vector: list[float]  # This needs to be added in by the script
    position_id: int  # This needs to be added in by the script
    activations: dict[str, Any]
    hard_negatives: list[dict[str, Any]]


class SAEVerlData(BaseModel):
    sae_id: int
    feature_vector: list[float]  # This needs to be added in by the script
    position_id: int  # This needs to be added in by the script
    activations: SAEActivationsV2  # Sentences that should activate the feature
    hard_negatives: list[SAEActivationsV2]  # Sentences that should NOT activate the feature

    @classmethod
    def from_typed_dict(cls, sae_data: "SAEVerlDataTypedDict") -> "SAEVerlData":
        return SAEVerlData.model_validate(sae_data)

    @classmethod
    def from_sae(cls, sae: SAEV2, feature_vector: list[float], position_id: int) -> "SAEVerlData":
        return SAEVerlData(
            sae_id=sae.sae_id,
            feature_vector=feature_vector,
            position_id=position_id,
            activations=sae.activations,
            hard_negatives=sae.hard_negatives,
        )


def make_sae_verl_typed_dict(sae_data: SAEV2, position_id: int, feature_vector: list[float]) -> SAEVerlDataTypedDict:
    return {
        "sae_id": sae_data.sae_id,
        "position_id": position_id,
        "feature_vector": feature_vector,
        "activations": sae_data.activations.model_dump(),
        "hard_negatives": [m.model_dump() for m in sae_data.hard_negatives],
    }

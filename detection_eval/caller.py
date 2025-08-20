import hashlib
from json import JSONDecodeError
import math
import time
import anthropic
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Mapping, Optional, Sequence, Type, TypeVar
from anthropic.types.message import Message
from anyio import Path as AnyioPath

import anyio
from dotenv import load_dotenv
from openai import NOT_GIVEN, AsyncOpenAI, InternalServerError
import os
from openai.types.moderation_create_response import ModerationCreateResponse
from pydantic import BaseModel, ValidationError
from slist import Slist
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from dataclasses import dataclass
import random
import openai


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


class ChatMessage(BaseModel):
    role: str
    content: str
    # base64
    image_content: str | None = None
    image_type: str | None = None  # image/jpeg, or image/png

    def as_text(self) -> str:
        return f"{self.role}:\n{self.content}"

    def to_openai_content(self) -> dict:
        if not self.image_content:
            return {
                "role": self.role,
                "content": self.content,
            }
        else:
            assert self.image_type, "Please provide an image type"
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{self.image_type};base64,{self.image_content}"},
                    },
                ],
            }

    def to_anthropic_content(self) -> dict:
        if not self.image_content:
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                ],
            }
        else:
            return {
                "role": self.role,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self.image_type or "image/jpeg",
                            "data": self.image_content,
                        },
                    },
                ],
            }


class ChatHistory(BaseModel):
    messages: Sequence[ChatMessage] = []

    @staticmethod
    def from_system(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="system", content=content)])

    @staticmethod
    def from_user(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="user", content=content)])

    @staticmethod
    def from_maybe_system(content: str | None) -> "ChatHistory":
        # Sometimes system prompt is optional in user functions.
        if content is None:
            return ChatHistory()
        else:
            return ChatHistory.from_system(content)

    def all_assistant_messages(self) -> Slist[ChatMessage]:
        return Slist(self.messages).filter(lambda msg: msg.role == "assistant")

    def as_text(self) -> str:
        return "\n".join([msg.as_text() for msg in self.messages])

    def add_user(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [ChatMessage(role="user", content=content)]
        return ChatHistory(messages=new_messages)

    def add_assistant(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [ChatMessage(role="assistant", content=content)]
        return ChatHistory(messages=new_messages)

    def add_messages(self, messages: Sequence[ChatMessage]) -> "ChatHistory":
        new_messages = list(self.messages) + list(messages)
        return ChatHistory(messages=new_messages)


class OpenaiResponse(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str | None = None
    system_fingerprint: str | None = None
    prompt_used: ChatHistory | None = None

    @property
    def first_response(self) -> str:
        try:
            content = self.choices[0]["message"]["content"]
            if content is None:
                raise ValueError(f"No content found in OpenaiResponse: {self}")
            return content
        except TypeError:
            raise ValueError(f"No content found in OpenaiResponse: {self}")

    @property
    def all_responses(self) -> list[str]:
        return [choice["message"]["content"] for choice in self.choices]

    @property
    def reasoning_content(self) -> str:
        ## sometimes has reasoning_content or reasoning instead of content e.g. deepseek-reasoner or gemini
        possible_keys = ["reasoning_content", "reasoning"]
        for key in possible_keys:
            if self.choices[0]["message"].get(key):
                return self.choices[0]["message"][key]
        raise ValueError(f"No reasoning_content found in OpenaiResponse: {self}")

    @property
    def has_reasoning(self) -> bool:
        possible_keys = ["reasoning_content", "reasoning"]
        for key in possible_keys:
            if self.choices[0]["message"].get(key):
                return True
        return False

    def has_response(self) -> bool:
        if len(self.choices) == 0:
            return False
        first_choice = self.choices[0]
        if first_choice["message"] is None:
            return False
        if first_choice["message"]["content"] is None:
            return False
        return True

    @property
    def hit_content_filter(self) -> bool:
        first_choice = self.choices[0]
        if "finishReason" in first_choice:
            if first_choice["finishReason"] == "content_filter":
                return True
        if "finish_reason" in first_choice:
            if first_choice["finish_reason"] == "content_filter":
                return True
        return False


class FileCacheRow(BaseModel):
    key: str
    response: str  # Should be generic, but w/e


class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]

    @property
    def single_response(self) -> str:
        if len(self.raw_responses) != 1:
            raise ValueError(f"This response has multiple responses {self.raw_responses}")
        else:
            return self.raw_responses[0]


class Prob(BaseModel):
    token: str
    prob: float


class LogProb(BaseModel):
    token: str
    logprob: float

    @property
    def proba(self) -> float:
        return math.exp(self.logprob)

    def to_prob(self) -> Prob:
        return Prob(token=self.token, prob=self.proba)


class TokenWithLogProbs(BaseModel):
    token: str
    logprob: float  # log probability of the particular token
    top_logprobs: Sequence[LogProb]  # log probability of the top 5 tokens

    def sorted_logprobs(self) -> Sequence[LogProb]:  # Highest to lowest
        return sorted(self.top_logprobs, key=lambda x: x.logprob, reverse=True)

    def sorted_probs(self) -> Sequence[Prob]:
        return [logprob.to_prob() for logprob in self.sorted_logprobs()]


class ResponseWithLogProbs(BaseModel):
    response: str
    content: Sequence[TokenWithLogProbs]  #


def write_jsonl_file_from_basemodel(path: Path | str, basemodels: Sequence[BaseModel]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json() + "\n")


def read_jsonl_file_into_basemodel(path: Path | str, basemodel: Type[GenericBaseModel]) -> Slist[GenericBaseModel]:
    with open(path) as f:
        return Slist(basemodel.model_validate_json(line) for line in f)


def deterministic_hash(something: str) -> str:
    return hashlib.sha1(something.encode()).hexdigest()


def validate_json_item(item: str, model: Type[GenericBaseModel]) -> GenericBaseModel | None:
    try:
        return model.model_validate_json(item)
    except ValidationError:
        print(f"Error validating {item} with model {model}")
        return None


class ToolArgs(BaseModel):
    tools: Sequence[Mapping[Any, Any]]
    tool_choice: str


class NotGivenSentinel:
    pass


NOT_GIVEN_SENTINEL = NotGivenSentinel()


class InferenceConfig(BaseModel):
    # todo: consider switching to NOT_GIVEN_SENTINEL instead of None
    # Config for openai
    model: str
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    # legacy APIs
    max_tokens: int | None = None
    # newer APIs that prefer the completion limits
    max_completion_tokens: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    # "minimal", "low", "medium", "high" for openai
    reasoning_effort: str | None = None
    continue_final_message: bool | None = None  # For runpod configs
    extra_body: dict | None = None

    def copy_update(
        self,
        temperature: float | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        top_p: float | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        max_tokens: int | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        max_completion_tokens: int | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        frequency_penalty: float | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        presence_penalty: float | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        n: int | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        continue_final_message: bool | NotGivenSentinel = NOT_GIVEN_SENTINEL,
        reasoning_effort: str | NotGivenSentinel = NOT_GIVEN_SENTINEL,
    ) -> "InferenceConfig":
        return InferenceConfig(
            model=self.model,
            temperature=temperature if not isinstance(temperature, NotGivenSentinel) else self.temperature,
            top_p=top_p if not isinstance(top_p, NotGivenSentinel) else self.top_p,
            max_tokens=max_tokens if not isinstance(max_tokens, NotGivenSentinel) else self.max_tokens,
            max_completion_tokens=max_completion_tokens
            if not isinstance(max_completion_tokens, NotGivenSentinel)
            else self.max_completion_tokens,
            frequency_penalty=frequency_penalty
            if not isinstance(frequency_penalty, NotGivenSentinel)
            else self.frequency_penalty,
            presence_penalty=presence_penalty
            if not isinstance(presence_penalty, NotGivenSentinel)
            else self.presence_penalty,
            n=n if not isinstance(n, NotGivenSentinel) else self.n,
            continue_final_message=continue_final_message
            if not isinstance(continue_final_message, NotGivenSentinel)
            else self.continue_final_message,
            reasoning_effort=reasoning_effort
            if not isinstance(reasoning_effort, NotGivenSentinel)
            else self.reasoning_effort,
        )


class OpenaiResponseWithLogProbs(BaseModel):
    choices: list[dict]
    usage: dict
    created: int
    model: str
    id: str
    system_fingerprint: str | None = None

    @property
    def first_response(self) -> str:
        return self.choices[0]["message"]["content"]

    def response_with_logprobs(self) -> ResponseWithLogProbs:
        response = self.first_response
        logprobs = self.choices[0]["logprobs"]["content"]
        parsed_content = [TokenWithLogProbs.model_validate(token) for token in logprobs]
        return ResponseWithLogProbs(response=response, content=parsed_content)

    def first_token_probability_for_target(self, target: str) -> float:
        logprobs = self.response_with_logprobs().content
        first_token = logprobs[0]
        for token in first_token.top_logprobs:
            # print(f"Token: {token.token} Logprob: {token.logprob}")
            if token.token == target:
                token_logprob = token.logprob
                # convert natural log to prob
                return math.exp(token_logprob)
        return 0.0


def file_cache_key(
    messages: ChatHistory,
    config: InferenceConfig,
    try_number: int,
    other_hash: str,
    tools: ToolArgs | None,
) -> str:
    messages_dump = messages.model_dump_json(exclude_none=True)
    config_dump = config.model_dump_json(exclude_none=True)  # for backwards compatibility
    tools_json = tools.model_dump_json(exclude_none=True) if tools is not None else ""  # for backwards compatibility
    _str = messages_dump + config_dump + tools_json + str(try_number) + other_hash
    return deterministic_hash(_str)


async def read_jsonl_file_into_basemodel_async(
    path: AnyioPath, basemodel: Type[GenericBaseModel]
) -> Slist[GenericBaseModel]:
    async with await anyio.open_file(path, "r") as f:
        return Slist([basemodel.model_validate_json(line) for line in await f.readlines()])


class Caller(ABC):
    @abstractmethod
    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        pass

    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        # todo: Not implemented for all callers.
        # yes this breaks liskov but too bad
        raise NotImplementedError()

    async def call_with_log_probs(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
    ) -> OpenaiResponseWithLogProbs:
        raise NotImplementedError()

    @abstractmethod
    async def flush(self) -> None:
        # flush file buffers
        raise NotImplementedError()

    ## implement context manager
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.flush()


class APIRequestCache(Generic[GenericBaseModel]):
    def __init__(self, cache_path: Path | str, response_type: Type[GenericBaseModel]):
        self.cache_path = AnyioPath(cache_path)
        self.response_type = response_type
        self.data: dict[str, str] = {}
        self.file_handler: Optional[anyio.AsyncFile] = None
        self.loaded_cache: bool = False
        self.cache_check_semaphore = anyio.Semaphore(1)

    async def flush(self) -> None:
        if self.file_handler:
            await self.file_handler.flush()

    async def load_cache(self) -> None:
        if await self.cache_path.exists():
            time_start = time.time()
            rows: Slist[FileCacheRow] = await read_jsonl_file_into_basemodel_async(
                path=self.cache_path,  # todo: asyncify
                basemodel=FileCacheRow,
            )
            time_end = time.time()
            n_items = len(rows)
            time_diff_1dp = round(time_end - time_start, 1)
            print(f"Loaded {n_items} items from {self.cache_path.as_posix()} in {time_diff_1dp} seconds")
        else:
            rows = Slist()
        for row in rows:
            self.data[row.key] = row.response
        self.loaded_cache = True

    async def get_file_handler(self) -> anyio.AsyncFile:
        if self.file_handler is None:
            # if the file doesn't exist, create it
            if not await self.cache_path.exists():
                # make parent directories
                await self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                # make sure it's created
                await self.cache_path.touch()
            self.file_handler = await anyio.open_file(self.cache_path, "a")
        return self.file_handler

    async def add_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int,
        response: GenericBaseModel,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> None:
        key = file_cache_key(messages, config, try_number, other_hash, tools=tools)
        response_str = response.model_dump_json()
        self.data[key] = response_str
        await self.write_line(key=key, response_json=response_str)

    async def get_model_call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int,
        tools: ToolArgs | None,
        other_hash: str = "",
    ) -> Optional[GenericBaseModel]:
        if not self.loaded_cache:
            async with self.cache_check_semaphore:
                # check again
                if not self.loaded_cache:
                    await self.load_cache()
        key = file_cache_key(messages, config, try_number, other_hash, tools=tools)
        response_str = self.data.get(key)
        if response_str:
            try:
                response = self.response_type.model_validate_json(response_str)
                # add the prompt used to the response
                return response
            except ValidationError as e:
                print(f"Warning: Failed to validate cache entry for key {key}")
                raise e
                # return None
        return None

    async def write_line(self, key: str, response_json: str) -> None:
        if not self.file_handler:
            await self.get_file_handler()
        if self.file_handler:
            # prevent multiple writes to same file
            async with self.cache_check_semaphore:
                line = FileCacheRow(key=key, response=response_json).model_dump_json() + "\n"
                await self.file_handler.write(line)


class CallerCache(Generic[GenericBaseModel]):
    """Will create a jsonl cache for each model."""

    def __init__(self, cache_path: Path, cache_type: Type[GenericBaseModel] = OpenaiResponse):
        self.cache_path = Path(cache_path)
        # if not exists, create it
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)
        assert self.cache_path.is_dir(), f"cache_path must be a folder, you provided {cache_path}"
        self.cache: dict[str, APIRequestCache[GenericBaseModel]] = {}
        self.log_probs_cache: dict[str, APIRequestCache[OpenaiResponseWithLogProbs]] = {}
        self.cache_type = cache_type

    def get_cache(self, model: str) -> APIRequestCache[GenericBaseModel]:
        if model not in self.cache:
            path = self.cache_path / f"{model}.jsonl"
            self.cache[model] = APIRequestCache(cache_path=path, response_type=self.cache_type)
        return self.cache[model]

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        if model not in self.log_probs_cache:
            path = self.cache_path / f"{model}_log_probs.jsonl"
            self.log_probs_cache[model] = APIRequestCache(cache_path=path, response_type=OpenaiResponseWithLogProbs)
        return self.log_probs_cache[model]

    async def flush(self) -> None:
        for cache in self.cache.values():
            await cache.flush()


class OpenAICaller(Caller):
    def __init__(
        self,
        cache_path: Path | str | CallerCache,
        api_key: str | None = None,
        organization: str | None = None,
        openai_client: AsyncOpenAI | None = None,
    ):
        if openai_client is not None:
            self.client = openai_client
        else:
            if api_key is None:
                env_key = os.getenv("OPENAI_API_KEY")
                assert env_key is not None, (
                    "Please provide an OpenAI API Key. Either pass it as an argument or set it in the environment variable OPENAI_API_KEY"
                )
                api_key = env_key
            self.client = AsyncOpenAI(api_key=api_key, organization=organization)
        self.cache_by_model = (
            CallerCache(Path(cache_path), cache_type=OpenaiResponse)
            if not isinstance(cache_path, CallerCache)
            else cache_path
        )

    async def flush(self) -> None:
        await self.cache_by_model.flush()

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        return self.cache_by_model.get_cache(model)

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        return self.cache_by_model.get_log_probs_cache(model)

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(retry_if_exception_type((ValidationError, JSONDecodeError, InternalServerError))),
        reraise=True,
    )
    @retry(
        stop=(stop_after_attempt(10)),
        wait=(wait_fixed(30)),  # for rate limits, wait longer
        retry=(retry_if_exception_type((openai.RateLimitError))),
        reraise=True,
    )
    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return maybe_result

        assert len(messages.messages) > 0, "Messages must be non-empty"
        extra_body = config.extra_body or {}
        if config.continue_final_message:
            # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
            # disable add_generation_prompt to continue the conversation
            extra_body["continue_final_message"] = config.continue_final_message
            extra_body["add_generation_prompt"] = not config.continue_final_message
        try:
            chat_completion = await self.client.chat.completions.create(
                model=config.model,
                messages=[msg.to_openai_content() for msg in messages.messages],  # type: ignore
                temperature=config.temperature if config.temperature is not None else NOT_GIVEN,
                max_tokens=config.max_tokens if config.max_tokens is not None else NOT_GIVEN,
                max_completion_tokens=(
                    config.max_completion_tokens if config.max_completion_tokens is not None else NOT_GIVEN
                ),
                top_p=config.top_p if config.top_p is not None else NOT_GIVEN,
                frequency_penalty=config.frequency_penalty if config.frequency_penalty != 0.0 else NOT_GIVEN,
                tools=tool_args.tools if tool_args is not None else NOT_GIVEN,  # type: ignore
                extra_body=extra_body or None,
                n=config.n,
                reasoning_effort=config.reasoning_effort if config.reasoning_effort is not None else NOT_GIVEN,  # type: ignore
            )
        except Exception as e:
            note = f"Model: {config.model}. API domain: {self.client.base_url}"
            e.add_note(note)
            raise e

        try:
            resp = OpenaiResponse.model_validate(chat_completion.model_dump())
        except ValidationError as e:
            print(
                f"Validation error for model {config.model}. Prompt: {messages}. resp: {chat_completion.model_dump()}"
            )
            raise e

        await self.get_cache(config.model).add_model_call(
            messages=messages, config=config, try_number=try_number, response=resp, tools=tool_args
        )
        return resp

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(
            retry_if_exception_type((ValidationError, JSONDecodeError, openai.RateLimitError, openai.APITimeoutError))
        ),
        reraise=True,
    )
    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> GenericBaseModel:
        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return schema.model_validate_json(maybe_result.first_response)
        try:
            chat_completion = await self.client.beta.chat.completions.parse(
                model=config.model,
                messages=[msg.to_openai_content() for msg in messages.messages],  # type: ignore
                temperature=config.temperature if config.temperature is not None else NOT_GIVEN,
                max_tokens=config.max_tokens if config.max_tokens is not None else NOT_GIVEN,
                max_completion_tokens=(
                    config.max_completion_tokens if config.max_completion_tokens is not None else NOT_GIVEN
                ),
                top_p=config.top_p if config.top_p is not None else NOT_GIVEN,
                frequency_penalty=config.frequency_penalty if config.frequency_penalty != 0.0 else NOT_GIVEN,
                response_format=schema,
                extra_body=config.extra_body or {},
                reasoning_effort=config.reasoning_effort if config.reasoning_effort is not None else NOT_GIVEN,  # type: ignore
                n=config.n if config.n is not None else NOT_GIVEN,
            )
        except Exception as e:
            api_key = self.client.api_key
            api_domain = self.client.base_url
            note = f"Model: {config.model}. API key: {api_key}. API domain: {api_domain}"
            e.add_note(note)
            raise e
        resp = OpenaiResponse.model_validate(chat_completion.model_dump())
        await self.get_cache(config.model).add_model_call(
            messages=messages, config=config, try_number=try_number, response=resp, tools=tool_args
        )
        return chat_completion.choices[0].message.parsed  # type: ignore

    async def call_with_log_probs(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        top_logprobs: int = 5,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponseWithLogProbs:
        maybe_result = await self.get_log_probs_cache(config.model).get_model_call(
            messages=messages, config=config, try_number=try_number, tools=tool_args, other_hash=str(top_logprobs)
        )
        if maybe_result is not None:
            return maybe_result

        result = await self.client.chat.completions.create(  # type: ignore
            model=config.model,
            messages=[msg.to_openai_content() for msg in messages.messages],  # type: ignore
            temperature=config.temperature if config.temperature is not None else NOT_GIVEN,
            max_tokens=config.max_tokens if config.max_tokens is not None else NOT_GIVEN,
            max_completion_tokens=(
                config.max_completion_tokens if config.max_completion_tokens is not None else NOT_GIVEN
            ),
            top_p=config.top_p if config.top_p is not None else NOT_GIVEN,
            frequency_penalty=config.frequency_penalty if config.frequency_penalty != 0.0 else NOT_GIVEN,
            n=config.n,
            stream=False,
            logprobs=True,
            top_logprobs=top_logprobs,
            reasoning_effort=config.reasoning_effort if config.reasoning_effort is not None else NOT_GIVEN,  # type: ignore
        )
        resp = OpenaiResponseWithLogProbs.model_validate(result.model_dump())

        await self.get_log_probs_cache(config.model).add_model_call(
            messages=messages,
            config=config,
            try_number=try_number,
            response=resp,
            other_hash=str(top_logprobs),
            tools=tool_args,
        )
        return resp


class AnthropicCaller(Caller):
    def __init__(
        self,
        cache_path: Path | str | CallerCache,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        api_key: str | None = None,
    ):
        if anthropic_client is not None:
            self.client = anthropic_client
        else:
            if api_key is None:
                env_key = os.getenv("ANTHROPIC_API_KEY")
                assert env_key is not None, "Please provide an Anthropic API Key"
                api_key = env_key
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.cache_by_model = CallerCache(Path(cache_path)) if not isinstance(cache_path, CallerCache) else cache_path

    async def flush(self) -> None:
        await self.cache_by_model.flush()

    def get_cache(self, model: str) -> APIRequestCache[OpenaiResponse]:
        return self.cache_by_model.get_cache(model)

    def get_log_probs_cache(self, model: str) -> APIRequestCache[OpenaiResponseWithLogProbs]:
        return self.cache_by_model.get_log_probs_cache(model)

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(retry_if_exception_type((ValidationError, anthropic.InternalServerError))),
        reraise=True,
    )
    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        assert tool_args is None, "Anthropic does not support tools"
        maybe_result = await self.get_cache(config.model).get_model_call(messages, config, try_number, tool_args)
        if maybe_result is not None:
            return maybe_result

        non_system, system = Slist(messages.messages).split_by(lambda msg: msg.role != "system")
        anthropic_messages = [{"role": msg.role, "content": msg.content} for msg in non_system]
        if system.length >= 2:
            raise ValueError("Anthropic does not support multiple system messages")
        system_message: ChatMessage | None = system.first_option
        to_pass_sys = system_message.content if system_message is not None else anthropic.NOT_GIVEN

        assert config.max_tokens is not None, "Anthropic requires max_tokens"
        response: Message = await self.client.messages.create(
            model=config.model,
            messages=anthropic_messages,  # type: ignore
            max_tokens=config.max_tokens,
            temperature=config.temperature if config.temperature is not None else anthropic.NOT_GIVEN,
            top_p=config.top_p if config.top_p is not None else anthropic.NOT_GIVEN,
            system=to_pass_sys,
        )
        # convert
        openai_response = OpenaiResponse(
            id=response.id,
            choices=[{"message": {"content": response.content[0].text, "role": "assistant"}}],  # type: ignore
            created=int(datetime.now().timestamp()),
            model=config.model,
            system_fingerprint=None,
            usage=response.usage.model_dump(),
        )

        await self.get_cache(config.model).add_model_call(
            messages=messages, config=config, try_number=try_number, response=openai_response, tools=tool_args
        )

        return openai_response

    @retry(
        stop=(stop_after_attempt(5)),
        wait=(wait_fixed(5)),
        retry=(retry_if_exception_type((ValidationError))),
        reraise=True,
    )
    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        raise NotImplementedError("Anthropic does not support schema parsing yet")

    async def call_with_log_probs(
        self, messages: ChatHistory, config: InferenceConfig, try_number: int = 1
    ) -> OpenaiResponseWithLogProbs:
        raise NotImplementedError("Anthropic does not support log probs yet")


@dataclass
class CallerConfig:
    name: str
    caller: Caller


class MultiClientCaller(Caller):
    def __init__(self, clients: Sequence[CallerConfig]):
        self.callers: list[CallerConfig] = list(clients)

    def merge(self, other: "MultiClientCaller") -> "MultiClientCaller":
        return MultiClientCaller(self.callers + other.callers)

    async def flush(self) -> None:
        for caller_config in self.callers:
            await caller_config.caller.flush()

    def _get_caller_for_model(self, model: str) -> Caller:
        # Router logic. It is simply a string match.
        for caller_config in self.callers:
            if caller_config.name in model:
                return caller_config.caller
        available_patterns = [caller_config.name for caller_config in self.callers]
        raise ValueError(f"No caller found for model {model}. Available patterns specified: {available_patterns}")

    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        caller = self._get_caller_for_model(config.model)
        return await caller.call(messages, config, try_number, tool_args)

    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        caller = self._get_caller_for_model(config.model)
        return await caller.call_with_schema(messages, schema, config, try_number)

    async def call_with_log_probs(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
    ) -> OpenaiResponseWithLogProbs:
        caller = self._get_caller_for_model(config.model)
        return await caller.call_with_log_probs(messages, config, try_number)


class PooledCaller(Caller):
    def __init__(self, callers: Sequence[Caller]):
        self.callers = callers

    async def flush(self) -> None:
        for caller in self.callers:
            await caller.flush()

    async def call(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        try_number: int = 1,
        tool_args: ToolArgs | None = None,
    ) -> OpenaiResponse:
        caller = random.choice(self.callers)
        return await caller.call(messages, config, try_number, tool_args)

    async def call_with_schema(
        self,
        messages: ChatHistory,
        schema: Type[GenericBaseModel],
        config: InferenceConfig,
        try_number: int = 1,
    ) -> GenericBaseModel:
        caller = random.choice(self.callers)
        return await caller.call_with_schema(messages, schema, config, try_number)

    async def call_with_log_probs(
        self, messages: ChatHistory, config: InferenceConfig, try_number: int = 1
    ) -> OpenaiResponseWithLogProbs:
        caller = random.choice(self.callers)
        return await caller.call_with_log_probs(messages, config, try_number)


class OpenAIModerateCaller:
    def __init__(self, api_key: str, cache_path: Path | str):
        self.api_key = api_key
        self.cache: APIRequestCache[ModerationCreateResponse] = APIRequestCache(
            cache_path=cache_path, response_type=ModerationCreateResponse
        )
        self.client = AsyncOpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(5),
        retry=retry_if_exception_type((ValidationError, InternalServerError)),
    )
    async def moderate(
        self,
        to_moderate: str,
        model: str = "omni-moderation-latest",
        try_number: int = 1,
    ) -> ModerationCreateResponse:
        """
        Moderates the given text using OpenAI's moderation API.

        Args:
            to_moderate (str): The text to be moderated.
            model (str): The model to use for moderation. Defaults to "omni-moderation-latest".
            try_number (int): The attempt number for retries. Defaults to 1.

        Returns:

            ModerationResponse: The parsed moderation response.

        """

        if self.cache is not None:
            maybe_result = await self.cache.get_model_call(
                messages=ChatHistory(messages=[ChatMessage(role="user", content=to_moderate)]),
                config=InferenceConfig(model=model),
                try_number=try_number,
                tools=None,
            )
            if maybe_result is not None:
                return maybe_result

        try:
            moderation_response: ModerationCreateResponse = await self.client.moderations.create(
                model=model,
                input=to_moderate,
            )

            # add the response to the cache
            if self.cache is not None:
                await self.cache.add_model_call(
                    messages=ChatHistory(messages=[ChatMessage(role="user", content=to_moderate)]),
                    config=InferenceConfig(model=model),
                    try_number=try_number,
                    response=moderation_response,
                    tools=None,
                )

            return moderation_response

        except ValidationError as ve:
            # Optionally, add logging here

            raise ve

        except Exception as e:
            # Optionally, handle other exceptions

            raise e


def load_openai_caller(cache_path: str | Path) -> OpenAICaller:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    shared_cache = CallerCache(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, cache_path=shared_cache)
    return openai_caller


def load_multi_caller(cache_path: str) -> MultiClientCaller:
    """Non-exhaustive list of models. For demonstration purposes.
    Simply copy and create a new function for your needs.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org = os.getenv("OPENAI_ORGANIZATION")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    assert anthropic_api_key, "Please provide an Anthropic API Key"

    assert openai_api_key, "Please provide an OpenAI API Key"
    assert openrouter_api_key, "Please provide an OpenRouter API Key"
    shared_cache = CallerCache(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, organization=openai_org, cache_path=shared_cache)
    openrouter_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        cache_path=shared_cache,
    )

    clients = [
        CallerConfig(name="gpt", caller=openai_caller),
        CallerConfig(name="google", caller=openrouter_caller),
        CallerConfig(name="qwen", caller=openrouter_caller),
        CallerConfig(name="deepseek", caller=openrouter_caller),
        CallerConfig(name="mistral", caller=openrouter_caller),
        CallerConfig(name="llama", caller=openrouter_caller),
        CallerConfig(
            name="claude",
            caller=AnthropicCaller(api_key=anthropic_api_key, cache_path=shared_cache),
        ),
    ]

    return MultiClientCaller(clients)


async def example_main():
    # Caches to the folder "cache"
    caller = load_openai_caller("cache")
    prompt = ChatHistory.from_user("How many letter 'r's are in the word 'strawberry?")
    config = InferenceConfig(temperature=1.0, max_tokens=100, model="gpt-4o")
    response = await caller.call(prompt, config)
    print(response.first_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_main())

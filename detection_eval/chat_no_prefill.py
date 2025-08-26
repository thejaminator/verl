"""Usage:
streamlit run detection_eval/chat_no_prefill.py
pip install streamlit-shortcuts==0.1.9
pip install streamlit
pip install openai
pip install python-dotenv
pip install transformers
"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_shortcuts import shortcut_button
from transformers import AutoTokenizer  # type: ignore

load_dotenv()


# Default values
DEFAULT_MAX_TOKENS = 6000
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_SYSTEM_PROMPT = ""


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "should_generate_response" not in st.session_state:
        st.session_state.should_generate_response = False
    if "editing_message_index" not in st.session_state:
        st.session_state.editing_message_index = None


def chat_with(model: str, url: str):
    api_key = os.environ.get("OW_DEFAULT_API_KEY")
    assert api_key is not None, "OW_DEFAULT_API_KEY is not set"
    client = OpenAI(base_url=url, api_key=api_key)

    return client, model


def clear_chat_history():
    st.session_state.messages = []


def retry_from_message(index):
    # Keep messages up to and including the selected user message
    st.session_state.messages = st.session_state.messages[: index + 1]
    # Set flag to generate a new response
    st.session_state.should_generate_response = True


def edit_message(index):
    # Set the index of the message being edited
    st.session_state.editing_message_index = index


def save_edited_message(index, new_content):
    # Update the message content
    st.session_state.messages[index]["content"] = new_content
    # Clear the editing state
    st.session_state.editing_message_index = None
    # Keep messages up to and including the edited user message
    st.session_state.messages = st.session_state.messages[: index + 1]
    # Set flag to generate a new response
    st.session_state.should_generate_response = True
    # No need to call retry_from_message as we've already done what it does


@st.cache_resource
def get_old_jinja_template() -> str:
    model = "unsloth/QwQ-32B"
    tokenizer = AutoTokenizer.from_pretrained(model)
    chat_template = tokenizer.chat_template
    return chat_template


def generate_response(model, api_messages, max_tokens, temperature, url, top_p, lora_adapter=None):
    client, model = chat_with(model, url)

    message_placeholder = st.empty()
    full_response = ""

    # Add system prompt if it exists
    messages_with_system = []
    if st.session_state.system_prompt.strip():
        messages_with_system.append({"role": "system", "content": st.session_state.system_prompt})
    messages_with_system.extend(api_messages)

    # Prepare extra_body with chat template and optional LoRA adapter
    extra_body = {}
    if lora_adapter and lora_adapter.strip():
        extra_body["lora_adapter"] = lora_adapter

    # Create the stream
    stream = client.chat.completions.create(
        model=model,
        messages=messages_with_system,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
    )

    # # Process the stream
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         full_response += chunk.choices[0].delta.content
    #         message_placeholder.markdown(full_response + "▌")

    # message_placeholder.markdown(full_response)
    message_placeholder.markdown(stream.choices[0].message.content)

    return stream.choices[0].message.content


def main(model: str, url: str):
    initialize_session_state()

    # Configuration section
    with st.sidebar:
        st.header("Model Configuration")

        # Display the current API URL (read-only)
        st.text(f"API URL: {url}")

        # System prompt text area
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=200,
            help="System message to set context and behavior for the AI assistant",
        )
        if system_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = system_prompt

        # LoRA Adapter input
        lora_adapter = st.text_input(
            "LoRA Adapter",
            value="",
            help="Optional LoRA adapter to use with the model (e.g., 'adapter_name' or path to adapter)",
        )

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=8000,
            value=DEFAULT_MAX_TOKENS,
            step=100,
            help="Maximum number of tokens to generate",
        )

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            help="Controls randomness: 0 is deterministic, higher values are more random",
        )

        # Top_p slider
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TOP_P,
            step=0.01,
            help="Controls diversity via nucleus sampling: 1.0 considers all tokens, lower values limit to more probable tokens",
        )

        # Add a clear button
        shortcut_button(
            "Clear Chat History (shortcut: left arrow)",
            on_click=clear_chat_history,
            help="Clear all messages in the chat ",
            shortcut="ArrowLeft",
        )

    shortcut_button(
        "Retry first message (shortcut: right arrow)",
        on_click=retry_from_message,
        args=(0,),
        help="Regenerate response from the first message",
        shortcut="ArrowRight",
    )

    # Display chat messages from history with retry buttons for user messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # For user messages, display the message and add a retry button below it
            if message["role"] == "user":
                # Check if this message is being edited
                if st.session_state.editing_message_index == i:
                    # Show an editable text area with the current message content
                    edited_content = st.text_area(
                        "Edit your message", value=message["content"], key=f"edit_textarea_{i}"
                    )
                    # Use buttons without columns
                    st.button("Save", key=f"save_edit_{i}", on_click=save_edited_message, args=(i, edited_content))
                    st.button(
                        "Cancel",
                        key=f"cancel_edit_{i}",
                        on_click=lambda: setattr(st.session_state, "editing_message_index", None),
                    )
                else:
                    # Display the message normally
                    st.markdown(message["content"])
                    # Always show buttons for user messages
                    st.button("Edit", key=f"edit_{i}", on_click=edit_message, args=(i,), help="Edit this message")

            else:
                st.markdown(message["content"])

    # Check if we need to generate a response (when the last message is from a user)
    if (
        len(st.session_state.messages) > 0
        and st.session_state.messages[-1]["role"] == "user"
        and (st.session_state.should_generate_response or len(st.session_state.messages) % 2 == 1)
    ):
        # Reset the flag
        st.session_state.should_generate_response = False

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Prepare messages for API call
            api_messages = []
            for msg in st.session_state.messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

            full_response = generate_response(model, api_messages, max_tokens, temperature, url, top_p, lora_adapter)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Get user input
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Prepare messages for API call
            api_messages = []
            for msg in st.session_state.messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            print(f"Sending messages: {api_messages}")

            full_response = generate_response(model, api_messages, max_tokens, temperature, url, top_p, lora_adapter)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    st.title("Chat with AI Model (LoRA)")

    # Get model ID input
    model_id = st.text_input(
        "Model ID",
        value="thejaminator/gemma-introspection-20250821",
        help="The ID of the model to use for chat",
    )

    # API URL for model loading
    url = st.text_input(
        "API URL", value="https://94nlcy6stx75yz-8000.proxy.runpod.net/v1", help="The base URL for the API"
    )

    main(model_id, url)

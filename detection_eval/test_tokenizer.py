# import tokenizer of Qwen3/Qwen3-8B
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# tokenize the following message
message = "X"

# print token id
token_ids = tokenizer.encode(message)
print(token_ids)

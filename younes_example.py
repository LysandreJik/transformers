#################
# This first part aims to use a transformers API to load a gguf file within it.
# It is somewhat working for tokenizers, but still needs to unquantize tensors before loading a model
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint_in_pytorch_model

# That's the file in question: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true
filename = "<path_to_file>/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

## Uncomment this if you want to load a tokenizer from the gguf file.
## Beware: it's NOT production-ready code at all, and mostly quick fixes just to see how the final API could look
## Do NOT consider this as finished code, as it will break in pretty much any other situation
## Additionally, there is a small issue with all spaces being replaced by the sentencepiece underline
# from transformers import LlamaTokenizerFast
# tok = LlamaTokenizerFast.from_pretrained(filename, from_gguf=True)

# This returns a dict of everything loaded from the gguf file.
# This doesn't dequantize the tensors, AFAIK
gguf_file = load_gguf_checkpoint_in_pytorch_model(filename)

###################
# This second part is based on https://github.com/99991/pygguf, and it is particularly relevant re. the
# unquantization of tensors
# The file is vendored for now, but if it is to be shipped, correct attribution should be given.

import transformers._gguf as gguf

with open(filename, "rb") as f:
    # Load metadata
    info, tensorinfo = gguf.load_gguf(f)

    # Print metadata
    for key, value in info.items():
        print(f"{key:30} {repr(value)[:100]}")

    print("gguf tensors")
    for key, value in tensorinfo.items():
        print(f"{key:30} {str(value)[:70]}")

    for name in tensorinfo:
        weights = gguf.load_gguf_tensor(f, tensorinfo, name)
        print(weights)

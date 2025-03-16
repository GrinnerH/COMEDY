import os
from transformers import LlamaTokenizer, LlamaForCausalLM

# 原始模型路径
model_path = "/home/grinner/Work_space/models/llama-2-7b"
# 转换后的保存路径
save_path = "/home/grinner/Work_space/models/llama-2-7b-hf"

# 加载并转换模型
model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=False,
    torch_dtype="auto",
    device_map="auto",
)

# 加载tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# 保存转换后的模型和tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
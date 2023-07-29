from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def belle_open_source_500k(data_file, tokenizer, max_len=2048):
    # https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/blob/main/Belle_open_source_0.5M.json
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < max_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= max_len:
            result["input_ids"][max_len - 1] = tokenizer.eos_token_id
            result["attention_mask"][max_len - 1] = 1

        result["labels"] = result["input_ids"].copy()
        # result = torch.tensor(result, dtype=torch.float16)
        return result

    def generate_and_tokenize_prompt(data_point):
        instruction = data_point['instruction']
        input_text = data_point["input"]
        input_text = "<|User|>:" + instruction + input_text + "<eoh>\n<|Bot|>:"
        target_text = data_point["output"] + "<eoa>\n"
        full_prompt = input_text + target_text
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_file)["train"]
    data = data.map(generate_and_tokenize_prompt, num_proc=16)
    return data

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True)
    ds = belle_open_source_500k("../data/Belle_open_source_0.5M.json", tokenizer, 512)
    print(ds[1])
    print(tokenizer.decode(ds[1]['input_ids']))
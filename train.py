import os
import torch
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from transformers import DataCollatorForSeq2Seq
from dataclasses import field, fields, dataclass
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb

from utils.model import load_model
from utils.dataset import belle_open_source_500k

# set target modules 
LINEAR_NAMES = ['gate_proj', 'q_proj', 'up_proj', 'o_proj', 'down_proj', 'v_proj', 'k_proj']

@dataclass
class FinetuneArguments:
    data_name: str = field()
    data_path: str = field()
    model_path: str = field(default="internlm/internlm-chat-7b")
    train_size: int = field(default=-1)
    test_size: int = field(default=200)
    max_len: int = field(default=2048)
    lora_rank: int = field(default=8)
    lora_modules: str = field(default=None)
    quantization: str = field(default='8bit')

def find_all_linear_names(model, quantization=None):
    """
    find out modules for LoraConfig

    Args:
        model (Any): model
        quantization (str, optional): quantization type, '4bit' or '8bit'. Defaults to None.

    Returns:
        List: modules name list
    """
    if not quantization: return []
    elif quantization=='8bit':
        cls = bnb.nn.Linear8bitLt
    elif quantization=='4bit':
        cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main():
    args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    
    set_seed(training_args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"world size {world_size} local rank {local_rank}")

    ### prepare model ###
    model, tokenizer = load_model(args.model_path, quantization=args.quantization, local_rank=local_rank)
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model, quantization=args.quantization)
    target_modules = args.lora_modules.split('.') if args.lora_modules is not None else modules

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    print(lora_config)
    model = get_peft_model(model, lora_config)

    ### prepare data ###
    data = eval(args.data_name)(args.data_path, tokenizer, args.max_len)
    if args.train_size > 0:
        data = data.shuffle(seed=training_args.seed).select(range(args.train_size))

    if args.test_size > 0:
        train_val = data.train_test_split(
            test_size=args.test_size, shuffle=True, seed=training_args.seed
        )
        train_data = train_val["train"].shuffle(seed=training_args.seed)
        val_data = train_val["test"].shuffle(seed=training_args.seed)
    else:
        train_data = data['train'].shuffle(seed=training_args.seed)
        val_data = None

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer,
                                             pad_to_multiple_of=8,
                                             return_tensors="pt",
                                             padding=True)
    )
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
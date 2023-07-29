from dataclasses import dataclass, field

@dataclass
class FinetuneArguments:
    data_name: str = field()
    data_path: str = field()
    model_path: str = field(default="internlm/internlm-chat-7b")
    train_size: int = field(default=-1) # 默认 -1，即使用全量数据
    test_size: int = field(default=200)
    max_len: int = field(default=2048)
    lora_rank: int = field(default=8)
    lora_modules: str = field(default=None)
    quantization: str = field(default='8bit')
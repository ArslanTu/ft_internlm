import bitsandbytes as bnb 

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
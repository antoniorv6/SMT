from smt_model.architectures.smt_arch import SMTModelForCausalLM
from smt_model.architectures.deepseek_arch import DeepSeekOCR2Wrapper
from smt_model.configuration_smt import SMTConfig

def build_model(config: SMTConfig, arch_type: str = "smt"):
    """
    Factory function to build either the original SMT architecture or the DeepSeek-OCR-2 architecture.
    """
    if arch_type == "smt":
        return SMTModelForCausalLM(config)
    elif arch_type == "deepseek":
        return DeepSeekOCR2Wrapper(config)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}. Expected 'smt' or 'deepseek'.")

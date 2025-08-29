from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.utils.peft import get_lora_model
from peft import LoraConfig, get_peft_model, TaskType
import torch
import types

def build_gr00t_finetune(pretrained_model_name_or_path="assets/weights/GR00T-N1.5-3B", tune_llm=True, **kwargs):
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        tune_llm=tune_llm,  # backbone's LLM
        tune_visual=True,  # backbone's vision tower
        tune_projector=True,  # action head's projector
        tune_diffusion_model=True,  # action head's DiT
    )
    
    

    if tune_llm:
        target_modules = []
        for name, module in model.backbone.named_modules():
            # Look for linear layers in attention mechanisms
            if isinstance(module, torch.nn.Linear):
                if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
                    target_modules.append(name)

        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        def _forward(self, inputs: dict):
            return self.base_model(inputs)
        model.backbone = get_peft_model(model.backbone, lora_config)
        model.backbone.prepare_input = model.backbone.base_model.prepare_input
        model.backbone.forward = types.MethodType(_forward, model.backbone)


    return model

MODEL_REPO = {
    'build_gr00t_finetune': build_gr00t_finetune
}

def create_model(model_name, **kwargs):
    return MODEL_REPO[model_name](**kwargs)
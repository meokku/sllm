import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# === 경로 설정 ===
BASE_MODEL_PATH = "/home/work/.deep_learning/skkullm/skku_sllm/models/base/Llama-3-Open-Ko-8B"
ADAPTER_PATH = "/home/work/.deep_learning/skkullm/skku_sllm/models/finetuned/skku-llm-20250505/checkpoint-4008_adapter_only"

# === 모델 로드 ===
print("✅ Base 모델 로드 중...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
print("✅ LoRA adapter 적용 모델 로드 중...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)

# === 파라미터 수 계산 ===
total_params = 0
lora_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if "lora_" in name:  # LoRA 관련 파라미터만 합산
        lora_params += param.numel()

# === 결과 출력 ===
print("\n📊 LoRA 파라미터 분석 결과")
print(f" - 전체 파라미터 수       : {total_params:,}")
print(f" - LoRA 학습 파라미터 수  : {lora_params:,}")
print(f" - LoRA 경량화 비율       : {100 * lora_params / total_params:.4f}%")

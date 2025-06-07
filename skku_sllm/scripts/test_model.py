import torch
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# 모델 경로 설정
BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = BASE_DIR / "models" / "base" / "Llama-3-Open-Ko-8B"
MODEL_ID = 'beomi/Llama-3-Open-Ko-8B'  # 원래 모델 ID (참조용)

print(f"모델 경로: {MODEL_PATH}")

# 모델 로드
print("모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",  # 자동으로 최적의 장치 할당
)
model.eval()

# 토크나이저 별도 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 파이프라인 설정 - device 파라미터 제거 (device_map="auto"와 충돌)
pipe = pipeline(
    'text-generation', 
    model=model,
    tokenizer=tokenizer,
)

def ask(x, context='', max_length=300, temperature=0.1, top_p=0.85):
    """
    질문에 대한 답변을 생성합니다.
    
    Args:
        x (str): 질문
        context (str): 추가 컨텍스트 또는 정보
        max_length (int): 생성할 최대 토큰 수
        temperature (float): 생성 다양성 조절 (낮을수록 일관성 높음)
        top_p (float): 누적 확률 기반 필터링 (낮을수록 일관성 높음)
    """
    # 프롬프트 구성
    prompt = f"### 질문: {x}\n\n### 맥락: {context}\n\n### 답변:" if context else f"### 질문: {x}\n\n### 답변:"
    
    # 생성 파라미터
    gen_kwargs = {
        "do_sample": True,
        "max_new_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "return_full_text": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,  # 반복 방지
        "no_repeat_ngram_size": 3,   # n-gram 반복 방지
    }
    
    # 텍스트 생성
    print(f"\n질문: {x}")
    if context:
        print(f"맥락: {context}")
    
    result = pipe(prompt, **gen_kwargs)
    answer = result[0]['generated_text'].strip()
    
    # 생성된 텍스트가 중간에 끊기거나 이상해지는 경우 정리
    # 일반적으로 '###'이나 특정 패턴 이후 등장하는 텍스트는 제거
    if "###" in answer:
        answer = answer.split("###")[0].strip()
    
    print(f"\n답변: {answer}")
    return answer

# 테스트 질문
print("\n=== 테스트 실행 ===")
ask("성균관대학교에서 학점 이수 기준은 어떻게 되나요?")

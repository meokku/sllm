#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import logging
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 로그 디렉토리 경로 설정
LOG_DIR = "/home/work/.deep_learning/skkullm/skku_sllm/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 기본 설정
BASE_MODEL_PATH = "/home/work/.deep_learning/skkullm/skku_sllm/models/base/Llama-3-Open-Ko-8B"
DATASET_PATH = "/home/work/.deep_learning/skkullm/skku_sllm/data/processed/skku_qa_transformed.jsonl"
OUTPUT_DIR = f"/home/work/.deep_learning/skkullm/skku_sllm/models/finetuned/skku-llm-{datetime.now().strftime('%Y%m%d')}"

# 학습 파라미터
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 1
EPOCHS = 4
LEARNING_RATE = 5e-5
CUTOFF_LEN = 2048
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.01
VAL_SET_SIZE = 0.1

# GPU 설정
def get_device_map():
    """
    사용 가능한 GPU 수에 따라 device_map을 구성합니다.
    """
    num_gpus = torch.cuda.device_count()
    logger.info(f"사용 가능한 GPU 수: {num_gpus}")
    
    if num_gpus == 0:
        return "cpu"
    elif num_gpus == 1:
        return "auto"
    else:
        # 단일 GPU만 사용하도록 설정 (멀티 GPU 사용 시 장치 간 텐서 이동 문제 방지)
        return {"": 0}  # 모든 모듈을 첫 번째 GPU에 할당

def load_jsonl_dataset(file_path):
    logger.info(f"데이터셋 로드 중: {file_path}")
    # Hugging Face datasets 라이브러리 사용하여 JSONL 파일 로드
    dataset = load_dataset('json', data_files=file_path)['train']
    logger.info(f"데이터셋 크기: {len(dataset)}")
    return dataset

def format_instruction(example):
    """
    지시문 형식으로 프롬프트를 포맷팅합니다.
    이제 input 필드는 비워두고, answer 필드만 사용합니다.
    """
    instruction = example['instruction']
    answer      = example['output']  # 기존 'output'

    # input 없이 instruction → answer 구조로만 프롬프트 생성
    prompt = f"<s>### 지시문:\n{instruction}\n\n### 응답:\n"
    return {
        "prompt": prompt,
        "response": answer
    }

def tokenize_function(examples, tokenizer, max_length):
    # 1) examples["prompt"], examples["response"] 은 batched=True 이면 리스트로 들어옵니다.
    prompts   = examples["prompt"]
    responses = examples["response"]

    # 2) 혹시라도 단일 문자열로 들어왔다면 리스트로 감싸기
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(responses, str):
        responses = [responses]

    # 3) None 또는 숫자 등이 섞여 있을 수 있으니 모두 문자열로 변환
    prompts   = ["" if p is None else str(p) for p in prompts]
    responses = ["" if r is None else str(r) for r in responses]

    # 4) 이제 tokenizer 에 넘기기
    prompt_tokens = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    response_tokens = tokenizer(
        responses,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,
    )

    # 5) 기존 로직 그대로
    input_ids = []
    attention_mask = []
    labels = []

    for p_ids, p_mask, r_ids, r_mask in zip(
        prompt_tokens["input_ids"],
        prompt_tokens["attention_mask"],
        response_tokens["input_ids"],
        response_tokens["attention_mask"],
    ):
        prompt_labels = [-100] * len(p_ids)
        combined_ids    = p_ids + r_ids[1:]
        combined_mask   = p_mask + r_mask[1:]
        combined_labels = prompt_labels + r_ids[1:]

        if len(combined_ids) > max_length:
            combined_ids    = combined_ids[:max_length]
            combined_mask   = combined_mask[:max_length]
            combined_labels = combined_labels[:max_length]

        input_ids.append(combined_ids)
        attention_mask.append(combined_mask)
        labels.append(combined_labels)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

def create_and_prepare_model(base_model_path):
    logger.info(f"모델 로드 중: {base_model_path}")
    
    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 장치 맵 가져오기
    device_map = get_device_map()
    logger.info(f"모델에 사용할 장치 맵: {device_map}")
    
    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    
    # 특수 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 오른쪽 패딩
    
    # EOS 토큰이 없는 경우 추가
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    
    # LoRA 설정
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    
    # 모델 준비
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def train():
    # 모델 및 토크나이저 준비
    model, tokenizer = create_and_prepare_model(BASE_MODEL_PATH)
    
    # 데이터셋 로드
    dataset = load_jsonl_dataset(DATASET_PATH)
    
    # 데이터 전처리
    processed_dataset = dataset.map(format_instruction)
    
    # 학습 및 검증 세트 분할
    if VAL_SET_SIZE > 0:
        processed_dataset = processed_dataset.train_test_split(test_size=VAL_SET_SIZE)
        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["test"]
    else:
        train_dataset = processed_dataset
        eval_dataset = None
    
    # 토큰화
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, CUTOFF_LEN),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, CUTOFF_LEN),
            batched=True,
            remove_columns=eval_dataset.column_names
        )
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    
    # 학습 인자 설정
    gradient_accumulation_steps = BATCH_SIZE // MICRO_BATCH_SIZE
    
    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        push_to_hub=False,
        # 단일 장치로 데이터 병렬 처리 비활성화
        local_rank=-1,
        ddp_find_unused_parameters=False,
        # 그라디언트 체크포인팅 활성화
        gradient_checkpointing=True,
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 학습 시작
    logger.info("학습 시작")
    trainer.train()
    
    # 모델 저장
    logger.info(f"모델 저장 중: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return OUTPUT_DIR

if __name__ == "__main__":
    # 저장 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 학습 실행
    output_dir = train()
    logger.info(f"파인튜닝 완료. 모델이 {output_dir}에 저장되었습니다.") 
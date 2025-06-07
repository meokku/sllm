#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
성균관대학교 QA 데이터셋 처리 스크립트

이 스크립트는 성균관대학교 관련 JSONL 형식의 QA 데이터셋을 처리하는 기능을 제공합니다.
- JSONL 파일 읽기
- 데이터셋 통계 확인
- 데이터 변환 및 전처리
"""

import json
import os
import pandas as pd
from pathlib import Path

def load_jsonl(file_path):
    """JSONL 파일을 로드하여 리스트로 반환합니다."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """데이터를 JSONL 형식으로 저장합니다."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def convert_to_instruction_format(data):
    """QA 데이터를 instruction 형식으로 변환합니다."""
    instruction_data = []
    for item in data:
        instruction_item = {
            "instruction": item["question"],
            "input": "",
            "output": item["answer"]
        }
        instruction_data.append(instruction_item)
    return instruction_data

def show_dataset_stats(data):
    """데이터셋 통계를 출력합니다."""
    print(f"데이터셋 크기: {len(data)}개 항목")
    
    question_lengths = [len(item["question"]) for item in data]
    answer_lengths = [len(item["answer"]) for item in data]
    
    print(f"질문 평균 길이: {sum(question_lengths)/len(question_lengths):.2f} 글자")
    print(f"답변 평균 길이: {sum(answer_lengths)/len(answer_lengths):.2f} 글자")
    print(f"질문 최소 길이: {min(question_lengths)} 글자")
    print(f"질문 최대 길이: {max(question_lengths)} 글자")
    print(f"답변 최소 길이: {min(answer_lengths)} 글자")
    print(f"답변 최대 길이: {max(answer_lengths)} 글자")

def main():
    # 경로 설정
    raw_data_path = Path("../data/raw/skku_qa_dataset.jsonl")
    processed_data_path = Path("../data/processed/skku_qa_instruction.jsonl")
    
    # 원본 데이터 로드
    print("JSONL 데이터 로드 중...")
    data = load_jsonl(raw_data_path)
    
    # 데이터셋 통계 확인
    print("\n데이터셋 통계:")
    show_dataset_stats(data)
    
    # instruction 형식으로 변환
    print("\n데이터를 instruction 형식으로 변환 중...")
    instruction_data = convert_to_instruction_format(data)
    
    # 처리된 데이터 저장
    print(f"변환된 데이터를 {processed_data_path}에 저장 중...")
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    save_jsonl(instruction_data, processed_data_path)
    
    print(f"변환 완료! 총 {len(instruction_data)}개 항목이 저장되었습니다.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from huggingface_hub import login, HfApi

def parse_args():
    parser = argparse.ArgumentParser(description='허깅페이스 허브에 모델 업로드')
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='업로드할 모델 경로'
    )
    
    parser.add_argument(
        '--repo_name', 
        type=str, 
        required=True,
        help='허깅페이스 저장소 이름 (예: skku/skku-llm)'
    )
    
    parser.add_argument(
        '--token', 
        type=str, 
        default=None,
        help='허깅페이스 토큰 (환경 변수로 설정하는 것을 권장)'
    )
    
    parser.add_argument(
        '--commit_message', 
        type=str, 
        default='성균관대학교 특화 LLM 업로드',
        help='커밋 메시지'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 토큰 설정
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("허깅페이스 토큰이 필요합니다. --token 인자 또는 HF_TOKEN 환경 변수를 설정해주세요.")
    
    # 허깅페이스 로그인
    login(token=token)
    
    # API 객체 생성
    api = HfApi()
    
    # 저장소가 존재하는지 확인하고, 없으면 생성
    try:
        api.repo_info(repo_id=args.repo_name)
        print(f"저장소 {args.repo_name}가 이미 존재합니다.")
    except Exception:
        api.create_repo(repo_id=args.repo_name, private=False)
        print(f"저장소 {args.repo_name}를 생성했습니다.")
    
    # 모델 업로드
    print(f"모델 업로드 중... ({args.model_path} -> {args.repo_name})")
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_name,
        commit_message=args.commit_message
    )
    
    print(f"모델이 성공적으로 업로드되었습니다: https://huggingface.co/{args.repo_name}")

if __name__ == "__main__":
    main() 
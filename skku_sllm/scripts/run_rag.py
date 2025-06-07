#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = "/home/work/.deep_learning/skkullm/skku_sllm"
sys.path.append(project_root)

from src.training.rag_builder import RAGBuilder
from src.training.rag_system import RAGSystem

def main():
    # RAG 시스템 초기화
    rag_builder = RAGBuilder()
    rag_system = RAGSystem(rag_builder)
    
    # 대화 루프
    print("\n성균관대학교 RAG 챗봇을 시작합니다. (종료하려면 'exit' 또는 'quit' 입력)")
    print("="*50)
    
    while True:
        # 사용자 입력 받기
        query = input("\n질문: ").strip()
        
        # 종료 조건 확인
        if query.lower() in ['exit', 'quit', '종료']:
            print("\n대화를 종료합니다.")
            break
        
        # 빈 입력 처리
        if not query:
            continue
        
        # 응답 생성
        response = rag_system.generate_response(query)
        
        # 응답 출력
        print("\n답변:", response)
        print("="*50)

if __name__ == "__main__":
    main()
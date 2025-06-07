#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.preprocess_data import SKKUDataProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='성균관대학교 크롤링 데이터 전처리')
    
    parser.add_argument('--input-dir', type=str, default='data/raw/skku_data',
                       help='크롤링 데이터 디렉토리 (기본값: data/raw/skku_data)')
    
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='전처리 결과 저장 디렉토리 (기본값: data/processed)')
    
    parser.add_argument('--min-length', type=int, default=50,
                       help='최소 내용 길이 (기본값: 50)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"성균관대학교 크롤링 데이터 전처리를 시작합니다...")
    print(f"입력 디렉토리: {args.input_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    
    # 전처리 객체 생성
    processor = SKKUDataProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # 전처리 실행
    processor.process_all()
    
    print("전처리가 완료되었습니다!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
from datetime import datetime

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.skku_crawler import SKKUCrawler
from src.data_processing.preprocess_data import SKKUDataProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='성균관대학교 데이터 수집 및 전처리 파이프라인')
    
    # 크롤링 관련 인자
    parser.add_argument('--crawl-mode', type=str, default='dfs',
                       choices=['dfs', 'all', 'notices', 'events', 'departments', 'facilities', 'menu', 'page'],
                       help='크롤링 모드 선택 (기본값: dfs)')
    
    parser.add_argument('--start-url', type=str, default=None,
                       help='크롤링 시작 URL (기본값: https://www.skku.edu/skku/)')
    
    parser.add_argument('--menu-url', type=str, default=None,
                       help='메뉴 크롤링 URL (crawl-mode=menu인 경우 필수)')
    
    parser.add_argument('--page-url', type=str, default=None,
                       help='페이지 크롤링 URL (crawl-mode=page인 경우 필수)')
    
    parser.add_argument('--max-depth', type=int, default=3,
                       help='크롤링할 최대 깊이 (기본값: 3)')
    
    parser.add_argument('--category-limit', action='store_true',
                        help='같은 카테고리 내의 페이지만 크롤링')
    
    parser.add_argument('--forbidden-patterns', type=str, nargs='+', default=[],
                        help='크롤링 제외할 URL 패턴 (예: "/login" "/search")')
    
    # 디렉토리 관련 인자
    parser.add_argument('--raw-dir', type=str, default='data/raw/skku_data',
                       help='크롤링 결과 저장 디렉토리 (기본값: data/raw/skku_data)')
    
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='전처리 결과 저장 디렉토리 (기본값: data/processed)')
    
    # 파이프라인 제어 인자
    parser.add_argument('--skip-crawl', action='store_true',
                       help='크롤링 단계 건너뛰기 (이미 크롤링한 데이터가 있는 경우)')
    
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='전처리 단계 건너뛰기')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"성균관대학교 데이터 수집 및 전처리 파이프라인을 시작합니다...")
    start_time = datetime.now()
    
    # 디렉토리 생성
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # 1. 크롤링 단계
    if not args.skip_crawl:
        print("\n=== 1단계: 데이터 크롤링 ===")
        crawler = SKKUCrawler(output_dir=args.raw_dir)
        
        # 금지 패턴 추가
        for pattern in args.forbidden_patterns:
            crawler.add_forbidden_pattern(pattern)
        
        if args.crawl_mode == 'dfs':
            crawler.crawl_dfs(
                start_url=args.start_url,
                max_depth=args.max_depth,
                category_limit=args.category_limit
            )
        elif args.crawl_mode == 'menu':
            if not args.menu_url:
                print("오류: --menu-url 옵션이 필요합니다.")
                return
            
            crawler.crawl_menu(
                menu_url=args.menu_url,
                max_depth=args.max_depth
            )
        elif args.crawl_mode == 'page':
            if not args.page_url:
                print("오류: --page-url 옵션이 필요합니다.")
                return
            
            crawler.crawl_specific_page(
                page_url=args.page_url,
                max_depth=args.max_depth
            )
        elif args.crawl_mode == 'all':
            crawler.crawl_all()
        elif args.crawl_mode == 'notices':
            crawler.crawl_notices()
        elif args.crawl_mode == 'events':
            crawler.crawl_events()
        elif args.crawl_mode == 'departments':
            crawler.crawl_departments()
        elif args.crawl_mode == 'facilities':
            crawler.crawl_facilities()
    else:
        print("\n=== 1단계: 데이터 크롤링 (건너뜀) ===")
    
    # 2. 데이터 전처리 단계
    if not args.skip_preprocess:
        print("\n=== 2단계: 데이터 전처리 ===")
        processor = SKKUDataProcessor(
            input_dir=args.raw_dir,
            output_dir=args.processed_dir
        )
        processor.process_all()
    else:
        print("\n=== 2단계: 데이터 전처리 (건너뜀) ===")
    
    # 종료 시간 및 요약
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n=== 파이프라인 완료 ===")
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요 시간: {duration}")
    print("\n데이터 파이프라인이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main() 
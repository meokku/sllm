#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from datetime import datetime

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.skku_crawler import SKKUCrawler

def parse_args():
    parser = argparse.ArgumentParser(description='성균관대학교 웹사이트 크롤러')
    
    parser.add_argument('--mode', type=str, default='dfs',
                        choices=['dfs', 'all', 'notices', 'events', 'departments', 'facilities', 'menu', 'page'],
                        help='크롤링 모드 선택 (기본값: dfs)')
    
    parser.add_argument('--start-url', type=str, default=None,
                        help='크롤링 시작 URL (기본값: https://www.skku.edu/skku/)')
    
    parser.add_argument('--menu-url', type=str, default=None,
                        help='메뉴 크롤링 URL (mode=menu인 경우 필수)')
    
    parser.add_argument('--page-url', type=str, default=None,
                        help='페이지 크롤링 URL (mode=page인 경우 필수)')
    
    parser.add_argument('--max-depth', type=int, default=3,
                        help='크롤링할 최대 깊이 (기본값: 3)')
    
    parser.add_argument('--max-pages', type=int, default=None,
                        help='크롤링할 최대 페이지 수 (기본값: 제한 없음)')
    
    parser.add_argument('--category-limit', action='store_true',
                        help='같은 카테고리 내의 페이지만 크롤링')
    
    parser.add_argument('--forbidden-patterns', type=str, nargs='+', default=[],
                        help='크롤링 제외할 URL 패턴 (예: "/login" "/search")')
    
    parser.add_argument('--output-dir', type=str, default='data/raw/skku_data',
                        help='크롤링 결과 저장 디렉토리 (기본값: data/raw/skku_data)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"성균관대학교 웹사이트 크롤러를 시작합니다 (모드: {args.mode})")
    
    # 크롤러 객체 생성
    crawler = SKKUCrawler(output_dir=args.output_dir)
    
    # 금지 패턴 추가
    for pattern in args.forbidden_patterns:
        crawler.add_forbidden_pattern(pattern)
    
    # 시작 시간
    start_time = datetime.now()
    print(f"크롤링 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 선택한 모드에 따라 크롤링 실행
    if args.mode == 'dfs':
        crawler.crawl_dfs(
            start_url=args.start_url,
            max_depth=args.max_depth,
            category_limit=args.category_limit,
            max_pages=args.max_pages
        )
    elif args.mode == 'menu':
        if not args.menu_url:
            print("오류: --menu-url 옵션이 필요합니다.")
            return
        
        crawler.crawl_menu(
            menu_url=args.menu_url,
            max_depth=args.max_depth,
            max_pages=args.max_pages
        )
    elif args.mode == 'page':
        if not args.page_url:
            print("오류: --page-url 옵션이 필요합니다.")
            return
        
        crawler.crawl_specific_page(
            page_url=args.page_url,
            max_depth=args.max_depth,
            max_pages=args.max_pages
        )
    elif args.mode == 'all':
        crawler.crawl_all()
    elif args.mode == 'notices':
        crawler.crawl_notices()
    elif args.mode == 'events':
        crawler.crawl_events()
    elif args.mode == 'departments':
        crawler.crawl_departments()
    elif args.mode == 'facilities':
        crawler.crawl_facilities()
    
    # 종료 시간
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"크롤링 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요 시간: {duration}")
    print("크롤링이 완료되었습니다!")

if __name__ == "__main__":
    main() 
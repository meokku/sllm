#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class SKKUCrawler:
    """성균관대학교 웹사이트에서 정보를 크롤링하는 클래스"""
    
    def __init__(self, base_url="https://www.skku.edu/skku/", output_dir="data/raw/skku_data", max_pages=100, max_depth=3):
        self.base_url = base_url
        self.output_dir = output_dir
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        }
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 셀레니움 웹드라이버 설정
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 헤드리스 모드
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
        except Exception as e:
            print(f"웹드라이버 초기화 중 오류 발생: {e}")
            print("셀레니움 기능은 사용할 수 없습니다.")
            self.driver = None
        
        # 금지 URL 패턴 (크롤링하지 않을 URL 패턴)
        self.forbidden_patterns = [
            # 공지사항 관련 패턴
            "/skk_comm/notice01.do",
            "notice01.do",
            "/skk_comm/news.do",
            "news.do",
            
            # 로그인/로그아웃/회원정보
            "/login", 
            "/logout", 
            "/member", 
            "/search",
            "/eng/",
            "/chi/",
            "/sitemap",
            "/news/notice",
            "/schedule",
            
            # 정적 파일 및 리소스
            "/images/",
            "/css/",
            "/js/",
            "/api/",
            "/download",
            "/popup",
            "/print",
            "/webzine",
            "/privacy",
            "/copyright",
            "/terms",
            ".pdf",
            ".hwp",
            ".xlsx",
            ".xls",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".jpg",
            ".png",
            ".gif",
            ".zip",
            "mailto:",
            "javascript:",
            "tel:"
        ]
        
        # 게시판 관련 URL 파라미터 패턴 (특정 상황에서만 체크)
        self.board_param_patterns = [
            "mode=edit",    # 편집 모드 URL
            "mode=write",   # 작성 모드 URL
            "mode=delete",  # 삭제 모드 URL
        ]
        
        # 게시판 네비게이션 텍스트 패턴
        self.board_nav_patterns = [
            "이전글", "다음글", "목록", "수정", "삭제", "글쓰기", "답글"
        ]
        
        # 허용하는 도메인 목록
        self.allowed_domains = ["www.skku.edu"]
        
        self.max_pages = max_pages
        self.max_depth = max_depth
        
    def __del__(self):
        """소멸자: 웹드라이버 종료"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()
    
    def add_forbidden_pattern(self, pattern):
        """크롤링 금지 URL 패턴 추가"""
        self.forbidden_patterns.append(pattern)
        print(f"크롤링 금지 패턴 추가: {pattern}")
    
    def get_soup(self, url):
        """지정된 URL에서 BeautifulSoup 객체 반환"""
        response = self.session.get(url, headers=self.headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    
    def save_to_json(self, data, filename):
        """데이터를 JSON 파일로 저장"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"데이터가 저장되었습니다: {filepath}")
        
    def save_to_csv(self, data, filename):
        """데이터를 CSV 파일로 저장"""
        filepath = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"데이터가 저장되었습니다: {filepath}")
    
    def save_to_jsonl(self, data, filename):
        """데이터를 JSONL 파일로 저장 (한 줄에 하나의 JSON 객체)"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"데이터가 저장되었습니다: {filepath}")
    
    def is_valid_url(self, url):
        """URL이 유효한지, skku.edu 도메인인지 확인"""
        if not url:
            return False
        
        # URL이 상대 경로인 경우 완전한 URL로 변환
        if url.startswith('/'):
            url = urljoin(self.base_url, url)
        
        # URL이 skku.edu 도메인인지 확인
        parsed_url = urlparse(url)
        return 'skku.edu' in parsed_url.netloc and parsed_url.scheme in ['http', 'https']
    
    def should_crawl(self, url):
        """URL이 크롤링 대상인지 확인 (금지 패턴 체크)"""
        # URL 파라미터 부분 분리 (? 이후)
        url_parts = url.split('?')
        url_path = url_parts[0]
        url_params = url_parts[1] if len(url_parts) > 1 else ""
        
        # 기본 금지 패턴 체크
        for pattern in self.forbidden_patterns:
            if pattern in url:
                print(f"  [건너뜀] 금지 패턴이 포함된 URL: {url}, 패턴: {pattern}")
                return False
        
        # URL이 공지사항 관련 경로인 경우에만 게시판 파라미터 패턴 체크
        if "notice" in url_path or "board" in url_path or "bbs" in url_path:
            # 게시판 파라미터 패턴 체크
            for pattern in self.board_param_patterns:
                if pattern in url_params:
                    print(f"  [건너뜀] 게시판 작업 URL: {url}, 패턴: {pattern}")
                    return False
                    
            # 게시판 네비게이션 텍스트 체크 (URL에 직접 포함된 경우)
            for pattern in self.board_nav_patterns:
                if pattern in url:
                    print(f"  [건너뜀] 게시판 내비게이션 링크: {url}, 패턴: {pattern}")
                    return False
                    
        return True
    
    def is_same_category(self, base_url, url):
        """두 URL이 동일한 카테고리(메뉴)에 속하는지 확인"""
        # 기본 URL과 대상 URL 파싱
        parsed_base = urlparse(base_url)
        parsed_url = urlparse(url)
        
        # 도메인이 다르면 다른 카테고리
        if parsed_base.netloc != parsed_url.netloc:
            return False
        
        # 경로(path)의 첫 번째 부분이 동일한지 확인
        base_path = parsed_base.path.strip('/').split('/')
        url_path = parsed_url.path.strip('/').split('/')
        
        # 경로 길이가 1 이상이면 첫 번째와 두 번째 경로 부분 비교
        # 예: /skku/about/과 /skku/about/pr/은 동일 카테고리
        if len(base_path) >= 2 and len(url_path) >= 2:
            return base_path[0] == url_path[0] and base_path[1] == url_path[1]
        
        return False
    
    def extract_content(self, url, soup):
        """웹 페이지에서 제목과 내용 추출"""
        title = ""
        content = ""
        
        print(f"\n[페이지 추출 시작] URL: {url}")
        
        # 불필요한 요소 제거
        self.remove_unnecessary_elements(soup)
        
        # 제목 추출 시도
        title_element_names = [
            'h1.page-title',
            'h1.title',
            'div.board-view-title',
            'h2.page-title',
            'h3.title',
            'title'
        ]
        
        title_elements = [soup.select_one(selector) for selector in title_element_names]
        
        for i, elem in enumerate(title_elements):
            if elem and elem.text.strip():
                title = elem.text.strip()
                print(f"  [제목 추출 성공] 선택자: {title_element_names[i]}, 제목: {title[:100]}")
                break
        
        if not title:
            print("  [제목 추출 실패] 제목 요소를 찾을 수 없습니다.")
        
        # 내용 추출 시도
        content_element_names = [
            'div.board-view-content',
            'div.content',
            'div.main-content',
            'div.article-content',
            'div.board-content'
        ]
        
        content_elements = [soup.select_one(selector) for selector in content_element_names]
        
        for i, elem in enumerate(content_elements):
            if elem and elem.text.strip():
                content = elem.text.strip()
                content_length = len(content)
                print(f"  [내용 추출 성공] 선택자: {content_element_names[i]}, 길이: {content_length}자")
                # 내용의 처음 100자와 마지막 100자만 출력
                if content_length > 200:
                    print(f"  [내용 미리보기] 처음 100자: {content[:100]}...")
                    print(f"  [내용 미리보기] 마지막 100자: ...{content[-100:]}")
                else:
                    print(f"  [내용 미리보기] {content}")
                break
        
        # 내용을 찾지 못한 경우 body 태그의 텍스트 사용
        if not content and soup.body:
            # 불필요한 요소가 제거된 body에서 텍스트 추출
            content = soup.body.text.strip()
            # 불필요한 공백 제거
            content = re.sub(r'\s+', ' ', content)
            print(f"  [내용 추출 대체] body 태그에서 내용 추출, 길이: {len(content)}자")
        
        if not content:
            print("  [내용 추출 실패] 내용 요소를 찾을 수 없습니다.")
        
        # 내용 정리: 연속된 공백, 줄바꿈 등을 정리
        content = self.clean_content(content)
        
        print(f"[페이지 추출 완료] 제목길이: {len(title)}자, 내용길이: {len(content)}자\n")
        
        return {
            "url": url,
            "title": title,
            "content": content
        }
    
    def remove_unnecessary_elements(self, soup):
        """불필요한 HTML 요소 제거"""
        # 제거할 요소의 CSS 선택자 목록
        selectors_to_remove = [
            # 네비게이션 메뉴
            'nav', '.nav', '.navigation', '.menu', '.gnb', '.lnb', '.sub-menu',
            # 검색창
            '.search', 'form', '.search-form', '.search-box', 'input[type="search"]',
            # 페이지네이션
            '.pagination', '.paging', '.page-navigation', '.page-numbers',
            # 헤더, 푸터
            'header', 'footer', '.header', '.footer', '#header', '#footer',
            # 사이드바
            'aside', '.sidebar', '.side-menu', '#sidebar',
            # 버튼
            'button', '.btn', '.button',
            # SNS 공유
            '.share', '.sns', '.social',
            # 웹 접근성 요소
            '.skip', '.skip-navigation', '.accessibility', '.a11y', 
            # 바로가기 링크
            'a[href="#jwxe_main_content"]', 'a[href="#jwxe_main_menu"]', 'a[href="#jwxe_sub_menu"]',
            # 기타 불필요한 요소
            '.banner', '.ad', '.advertisement', '#sitemap',
            '.quick-menu', '.utility', '.breadcrumb', '.copyright',
            'script', 'style', 'iframe', 'noscript'
        ]
        
        # 특정 요소의 ID나 클래스 목록
        ids_to_remove = ['search', 'pagination', 'menu', 'header', 'footer', 'sidebar']
        classes_to_contain = ['search', 'paging', 'page', 'menu', 'nav', 'header', 'footer', 'side']
        
        removed_count = 0
        
        # CSS 선택자로 요소 제거
        for selector in selectors_to_remove:
            elements = soup.select(selector)
            for element in elements:
                element.decompose()
                removed_count += 1
        
        # ID로 요소 제거
        for id_name in ids_to_remove:
            elements = soup.find_all(id=re.compile(id_name, re.I))
            for element in elements:
                element.decompose()
                removed_count += 1
        
        # 클래스명에 특정 문자열이 포함된 요소 제거
        for class_part in classes_to_contain:
            elements = soup.find_all(class_=lambda c: c and class_part in c.lower())
            for element in elements:
                element.decompose()
                removed_count += 1
        
        print(f"  [전처리] {removed_count}개의 불필요한 요소가 제거되었습니다.")
        return soup
    
    def clean_content(self, text):
        """내용 정리 및 정제"""
        if not text:
            return ""
            
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 여러 공백을 하나로 대체
        text = re.sub(r'\s+', ' ', text)
        
        # 숫자로만 이루어진 짧은 텍스트 제거 (페이지 번호 등)
        text = re.sub(r'\b\d{1,3}\b', '', text)
        
        # 웹 접근성 관련 텍스트 제거 (바로가기 등)
        accessibility_patterns = [
            '본문 바로가기', '주메뉴 바로가기', '서브메뉴 바로가기', 
            '메뉴 바로가기', '푸터 바로가기', '사이트맵 바로가기',
            '콘텐츠 바로가기', '하단메뉴 바로가기', '네비게이션 바로가기',
            '검색 바로가기', '링크 바로가기', '상단메뉴 바로가기',
            '상단으로', '맨위로', '맨 위로', '처음으로', '맨 처음으로'
        ]
        
        for pattern in accessibility_patterns:
            text = text.replace(pattern, '')
        
        # 불필요한 짧은 단어 제거 (검색, 이전, 다음 등)
        words_to_remove = ['검색', '이전', '다음', '처음', '마지막', '페이지', '작성자', '조회수', 
                          '등록일', '수정일', '첨부파일', '목록', '글쓰기', '답글', '삭제', 
                          '작성일', '담당부서', '담당자', '연락처', '이메일', 'TOP']
        
        for word in words_to_remove:
            text = re.sub(r'\b' + word + r'\b', '', text)
        
        # 공백 정리
        text = text.strip()
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def parse_links(self, base_url, soup):
        """웹 페이지에서 내부 링크 추출"""
        links = []
        
        try:
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                # 상대 URL을 절대 URL로 변환
                absolute_url = urljoin(base_url, href)
                
                # URL 정규화 (쿼리 스트링 제거, 프래그먼트 제거 등)
                parsed_url = urlparse(absolute_url)
                
                # 허용된 도메인인지 확인
                if parsed_url.netloc not in self.allowed_domains:
                    continue
                
                # 쿼리 스트링과 프래그먼트 제거
                normalized_url = urlunparse((
                    parsed_url.scheme,
                    parsed_url.netloc,
                    parsed_url.path,
                    parsed_url.params,
                    '',  # 쿼리 스트링 제거
                    ''   # 프래그먼트 제거
                ))
                
                if normalized_url not in links:
                    links.append(normalized_url)
        except Exception as e:
            print(f"링크 파싱 중 오류 발생: {e}")
        
        return links
    
    def crawl_dfs(self, start_url=None, max_depth=5, category_limit=False, max_pages=None):
        """DFS 알고리즘을 사용하여 성균관대학교 웹사이트 크롤링"""
        if not start_url:
            start_url = self.base_url
        
        print(f"\n===== DFS 알고리즘으로 크롤링을 시작합니다 =====")
        print(f"시작 URL: {start_url}")
        print(f"최대 깊이: {max_depth}")
        if max_pages:
            print(f"최대 페이지 수: {max_pages}")
        if category_limit:
            print(f"카테고리 제한 모드: 활성화 (같은 카테고리 내 페이지만 크롤링)")
        else:
            print(f"카테고리 제한 모드: 비활성화 (모든 페이지 크롤링)")
        
        if self.forbidden_patterns:
            print(f"크롤링 제외 패턴: {', '.join(self.forbidden_patterns)}")
        print("=================================================\n")
        
        # 방문한 URL을 저장할 집합
        visited = set()
        # 크롤링한 데이터를 저장할 리스트
        crawled_data = []
        # 탐색할 URL 스택 (URL, 깊이)
        stack = [(start_url, 0)]
        
        # 진행 상태 카운터
        page_count = 0
        
        # 데이터를 저장할 파일 준비
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        url_category = self.extract_category_name(start_url)
        filename = f"skku_{url_category}_{timestamp}.jsonl"
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"크롤링 데이터는 실시간으로 저장됩니다: {filepath}")
        
        while stack and (max_pages is None or page_count < max_pages):
            current_url, depth = stack.pop()
            
            # 이미 방문했거나 최대 깊이를 초과한 URL은 건너뜀
            if current_url in visited or depth > max_depth:
                continue
            
            # URL이 유효하지 않으면 건너뜀
            if not self.is_valid_url(current_url):
                print(f"[건너뜀] 유효하지 않은 URL: {current_url}")
                continue
            
            # 금지된 패턴이 있는 URL은 건너뜀
            if not self.should_crawl(current_url):
                print(f"[건너뜀] 금지 패턴이 포함된 URL: {current_url}")
                continue
            
            # 카테고리 제한 모드인 경우, 동일 카테고리 URL만 크롤링
            if category_limit and not self.is_same_category(start_url, current_url):
                print(f"[건너뜀] 다른 카테고리 URL: {current_url}")
                continue
            
            page_count += 1
            print(f"\n[크롤링 {page_count}] 깊이: {depth}, URL: {current_url}")
            
            try:
                # 웹 페이지 가져오기
                print(f"  웹 페이지 요청 중...")
                soup = self.get_soup(current_url)
                print(f"  웹 페이지 로드 완료")
                
                # 방문 표시
                visited.add(current_url)
                
                # 내용 추출
                page_data = self.extract_content(current_url, soup)
                
                # 로그인 페이지 체크 및 스킵
                is_login_page = False
                if page_data["title"] and any(title_pattern in page_data["title"] for title_pattern in [
                    "로그인", "성균관대 ( 대표 홈페이지 ) | 기타 | 로그인", "통합로그인"
                ]):
                    print(f"  [저장 안함] 로그인 관련 페이지입니다: {page_data['title']}")
                    is_login_page = True
                
                # 로그인 페이지가 아니고 내용이 의미 있는 경우에만 저장
                if not is_login_page and (page_data["title"] or len(page_data["content"]) > 100):
                    # 데이터를 리스트에 추가
                    crawled_data.append(page_data)
                    # 데이터를 즉시 파일에 추가 (한 줄씩 JSONL 형식으로)
                    with open(filepath, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                    print(f"  [저장 완료] 제목: {page_data['title'][:50] if page_data['title'] else '(제목 없음)'}")
                    print(f"  [실시간 저장] 데이터가 {filename}에 즉시 저장되었습니다.")
                elif not is_login_page:
                    print(f"  [저장 안함] 의미 있는 내용이 없습니다. (제목 길이: {len(page_data['title'])}, 내용 길이: {len(page_data['content'])})")
                
                # 모든 링크 찾기
                if depth < max_depth:
                    print(f"  링크 탐색 중...")
                    
                    # parse_links 함수를 사용하여 링크 추출
                    link_urls = self.parse_links(current_url, soup)
                    
                    # 링크를 스택에 추가
                    links = []
                    for link_url in link_urls:
                        # 유효한 URL이고 아직 방문하지 않았으며 금지 패턴이 없는 경우
                        if (self.is_valid_url(link_url) and 
                            link_url not in visited and
                            self.should_crawl(link_url)):
                            links.append((link_url, depth + 1))
                    
                    # 링크를 스택에 추가 (역순으로 추가하여 원래 순서대로 처리)
                    for link in reversed(links):
                        stack.append(link)
                    
                    print(f"  발견된 링크 수: {len(links)}, 스택에 추가됨")
                
                # 너무 빠른 요청으로 서버에 부하를 주지 않도록 지연
                print(f"  다음 요청까지 1초 대기...")
                time.sleep(1)
                
            except Exception as e:
                print(f"[오류] URL 크롤링 중 오류 발생: {current_url}")
                print(f"[오류] 메시지: {e}")
            
            # 최대 페이지 수 체크
            if max_pages and page_count >= max_pages:
                print(f"\n[알림] 최대 페이지 수({max_pages})에 도달했습니다. 크롤링을 종료합니다.")
                break
        
        print(f"\n===== 크롤링 완료 =====")
        print(f"총 {len(visited)}개 페이지 방문")
        print(f"총 {len(crawled_data)}개 데이터 저장")
        if max_pages and page_count >= max_pages:
            print(f"최대 페이지 수({max_pages})에 의해 크롤링이 제한되었습니다.")
        print(f"파일명: {filename}")
        print("========================\n")
        
        return crawled_data
    
    def extract_category_name(self, url):
        """URL에서 카테고리 이름 추출"""
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        # 경로가 2단계 이상인 경우
        if len(path_parts) >= 2:
            return path_parts[1]  # 두 번째 부분 (첫 번째는 보통 'skku')
        
        # 경로가 1단계 이하인 경우 또는 없는 경우
        return "main"
    
    def crawl_menu(self, menu_url, max_depth=3, max_pages=None):
        """특정 메뉴/카테고리 단위로 크롤링"""
        print(f"메뉴 크롤링을 시작합니다: {menu_url}")
        
        # 카테고리 제한 모드로 DFS 크롤링 실행
        return self.crawl_dfs(
            start_url=menu_url,
            max_depth=max_depth,
            category_limit=True,
            max_pages=max_pages
        )
    
    def crawl_specific_page(self, page_url, max_depth=5, max_pages=None):
        """특정 페이지와 그 페이지의 링크들을 크롤링"""
        print(f"특정 페이지 및 연결된 페이지 크롤링을 시작합니다: {page_url}")
        
        # 페이지 수 제한을 위한 카운터
        page_counter = {
            'count': 0,
            'max': max_pages
        }
        
        # 방문한 URL 집합과 데이터 리스트
        visited = set()
        crawled_data = []
        
        # 데이터를 저장할 파일 준비
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        url_parts = urlparse(page_url).path.strip('/').split('/')
        page_category = url_parts[-2] if len(url_parts) > 1 else "page"
        filename = f"skku_{page_category}_{timestamp}.jsonl"
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"크롤링 데이터는 실시간으로 저장됩니다: {filepath}")
        
        # DFS 방식으로 페이지를 크롤링하는 도우미 함수
        def crawl_recursive(url, depth, visited, data_list, counter):
            # 이미 방문했거나 최대 깊이를 초과한 경우 종료
            if url in visited or depth > max_depth:
                return
            
            # 최대 페이지 수에 도달한 경우 종료
            if counter['max'] and counter['count'] >= counter['max']:
                print(f"\n[알림] 최대 페이지 수({counter['max']})에 도달했습니다. 크롤링을 종료합니다.")
                return
            
            counter['count'] += 1
            print(f"\n[크롤링 {counter['count']}] 깊이: {depth}, URL: {url}")
            visited.add(url)
            
            try:
                # 웹 페이지 가져오기
                print(f"  웹 페이지 요청 중...")
                soup = self.get_soup(url)
                print(f"  웹 페이지 로드 완료")
                
                # 내용 추출
                page_data = self.extract_content(url, soup)
                
                # 로그인 페이지 체크 및 스킵
                is_login_page = False
                if page_data["title"] and any(title_pattern in page_data["title"] for title_pattern in [
                    "로그인", "성균관대 ( 대표 홈페이지 ) | 기타 | 로그인", "통합로그인"
                ]):
                    print(f"  [저장 안함] 로그인 관련 페이지입니다: {page_data['title']}")
                    is_login_page = True
                
                # 의미 있는 내용이고 로그인 페이지가 아닌 경우에만 저장
                if not is_login_page and (page_data["title"] or len(page_data["content"]) > 100):
                    # 데이터 리스트에 추가
                    data_list.append(page_data)
                    # 데이터를 즉시 파일에 추가 (한 줄씩 JSONL 형식으로)
                    with open(filepath, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                    print(f"  [저장 완료] 제목: {page_data['title'][:50] if page_data['title'] else '(제목 없음)'}")
                    print(f"  [실시간 저장] 데이터가 {filename}에 즉시 저장되었습니다.")
                
                # 다음 페이지로 이동할 링크 찾기
                if depth < max_depth:
                    print(f"  링크 탐색 중...")
                    
                    # parse_links 함수를 사용하여 링크 추출
                    link_urls = self.parse_links(url, soup)
                    
                    # 링크 필터링
                    links = []
                    for link_url in link_urls:
                        # 유효한 URL이고 아직 방문하지 않았으며 금지 패턴이 없는 경우
                        if (self.is_valid_url(link_url) and 
                            link_url not in visited and
                            self.should_crawl(link_url)):
                            links.append(link_url)
                    
                    print(f"  발견된 유효 링크 수: {len(links)}")
                    
                    # 찾은 모든 링크에 대해 재귀적으로 크롤링
                    for link in links:
                        print(f"  다음 링크로 이동: {link}")
                        # 요청 간 간격 두기
                        time.sleep(1)
                        crawl_recursive(link, depth + 1, visited, data_list, counter)
            
            except Exception as e:
                print(f"[오류] URL 크롤링 중 오류 발생: {url}")
                print(f"[오류] 메시지: {e}")
        
        # 재귀적으로 크롤링 실행
        crawl_recursive(page_url, 0, visited, crawled_data, page_counter)
        
        print(f"\n===== 크롤링 완료 =====")
        print(f"총 {len(visited)}개 페이지 방문")
        print(f"총 {len(crawled_data)}개 데이터 저장")
        if max_pages and page_counter['count'] >= max_pages:
            print(f"최대 페이지 수({max_pages})에 의해 크롤링이 제한되었습니다.")
        print(f"파일명: {filename}")
        print("========================\n")
        
        return crawled_data
    
    def crawl_notices(self, page_count=5):
        """공지사항 크롤링"""
        print("공지사항 크롤링을 시작합니다...")
        notices = []
        
        # 메인 공지사항 URL
        notice_url = urljoin(self.base_url, "about/skku_notice/notice_list.do")
        
        try:
            self.driver.get(notice_url)
            
            for page in range(1, page_count + 1):
                print(f"공지사항 페이지 {page} 크롤링 중...")
                
                # 페이지 로딩 대기
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".board-list-table"))
                )
                
                # 공지사항 목록 추출
                rows = self.driver.find_elements(By.CSS_SELECTOR, ".board-list-table tbody tr")
                
                for row in rows:
                    try:
                        # 공지 여부 확인 (공지는 notice 클래스가 있음)
                        is_notice = len(row.find_elements(By.CSS_SELECTOR, ".notice")) > 0
                        
                        # 제목과 링크 추출
                        title_elem = row.find_element(By.CSS_SELECTOR, "td.subject a")
                        title = title_elem.text.strip()
                        link = title_elem.get_attribute("href")
                        
                        # 날짜 추출
                        date = row.find_element(By.CSS_SELECTOR, "td.date").text.strip()
                        
                        notices.append({
                            "title": title,
                            "link": link,
                            "date": date,
                            "is_notice": is_notice
                        })
                    except Exception as e:
                        print(f"공지사항 항목 추출 중 오류: {e}")
                
                # 다음 페이지로 이동 (마지막 페이지가 아닌 경우)
                if page < page_count:
                    next_page_button = self.driver.find_element(By.CSS_SELECTOR, f".pagination a[href*='cpage={page+1}']")
                    next_page_button.click()
                    time.sleep(1)  # 페이지 로딩 대기
        
        except Exception as e:
            print(f"공지사항 크롤링 중 오류 발생: {e}")
        
        self.save_to_json(notices, "skku_notices.json")
        return notices
    
    def crawl_events(self):
        """학교 행사 정보 크롤링"""
        print("학교 행사 정보 크롤링을 시작합니다...")
        events = []
        
        # 학사일정 URL
        calendar_url = urljoin(self.base_url, "edu/handbook/ca_list.do")
        
        try:
            self.driver.get(calendar_url)
            
            # 페이지 로딩 대기
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".calendar-area"))
            )
            
            # 연도별 탭 선택 (현재 연도)
            current_year = datetime.now().year
            year_tabs = self.driver.find_elements(By.CSS_SELECTOR, ".ui-tab-list a")
            for tab in year_tabs:
                if str(current_year) in tab.text:
                    tab.click()
                    time.sleep(1)
                    break
            
            # 행사 정보 추출
            event_items = self.driver.find_elements(By.CSS_SELECTOR, ".calendar-area .calendar-list li")
            
            for item in event_items:
                try:
                    date = item.find_element(By.CSS_SELECTOR, ".date").text.strip()
                    title = item.find_element(By.CSS_SELECTOR, ".txt").text.strip()
                    
                    events.append({
                        "date": date,
                        "title": title
                    })
                except Exception as e:
                    print(f"행사 정보 추출 중 오류: {e}")
        
        except Exception as e:
            print(f"행사 정보 크롤링 중 오류 발생: {e}")
        
        self.save_to_json(events, "skku_events.json")
        return events
    
    def crawl_departments(self):
        """학과 정보 크롤링"""
        print("학과 정보 크롤링을 시작합니다...")
        departments = []
        
        # 단과대학 및 대학원 URL
        college_url = urljoin(self.base_url, "college/college_intro.do")
        
        try:
            self.driver.get(college_url)
            
            # 페이지 로딩 대기
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".college-list"))
            )
            
            # 단과대학 목록 추출
            college_items = self.driver.find_elements(By.CSS_SELECTOR, ".college-list li")
            
            for item in college_items:
                try:
                    name = item.find_element(By.CSS_SELECTOR, "strong").text.strip()
                    link = item.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    
                    # 이미지 URL 추출 (있는 경우)
                    try:
                        img_url = item.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
                    except:
                        img_url = None
                    
                    departments.append({
                        "name": name,
                        "link": link,
                        "img_url": img_url,
                        "type": "college"
                    })
                except Exception as e:
                    print(f"단과대학 정보 추출 중 오류: {e}")
        
        except Exception as e:
            print(f"학과 정보 크롤링 중 오류 발생: {e}")
        
        self.save_to_json(departments, "skku_departments.json")
        return departments
    
    def crawl_facilities(self):
        """학교 시설 정보 크롤링"""
        print("학교 시설 정보 크롤링을 시작합니다...")
        facilities = []
        
        # 캠퍼스 정보 URL
        campus_url = urljoin(self.base_url, "about/campusinfo/campus_intro.do")
        
        try:
            self.driver.get(campus_url)
            
            # 페이지 로딩 대기
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".campus-facility"))
            )
            
            # 캠퍼스 탭 (인문사회과학/자연과학) 목록
            campus_tabs = self.driver.find_elements(By.CSS_SELECTOR, ".ui-tab-list a")
            
            for i, tab in enumerate(campus_tabs):
                tab_name = tab.text.strip()
                print(f"{tab_name} 캠퍼스 시설 정보 크롤링 중...")
                
                # 탭 클릭
                self.driver.execute_script("arguments[0].click();", tab)
                time.sleep(1)
                
                # 시설 정보 추출
                facility_items = self.driver.find_elements(By.CSS_SELECTOR, ".campus-facility li")
                
                for item in facility_items:
                    try:
                        name = item.find_element(By.CSS_SELECTOR, "strong").text.strip()
                        desc = item.find_element(By.CSS_SELECTOR, "p").text.strip()
                        
                        facilities.append({
                            "name": name,
                            "description": desc,
                            "campus": tab_name
                        })
                    except Exception as e:
                        print(f"시설 정보 추출 중 오류: {e}")
        
        except Exception as e:
            print(f"학교 시설 정보 크롤링 중 오류 발생: {e}")
        
        self.save_to_json(facilities, "skku_facilities.json")
        return facilities
    
    def crawl_all(self):
        """모든 크롤링 함수 실행"""
        print("성균관대학교 데이터 전체 크롤링을 시작합니다...")
        
        # 기존 크롤링 함수 실행
        self.crawl_notices()
        self.crawl_events()
        self.crawl_departments()
        self.crawl_facilities()
        
        # DFS 크롤링 실행 (메인 사이트)
        self.crawl_dfs(self.base_url, max_depth=3)
        
        print("모든 크롤링이 완료되었습니다.")


# 실행 코드
if __name__ == "__main__":
    crawler = SKKUCrawler()
    
    # DFS 알고리즘으로 크롤링 실행
    crawler.crawl_dfs(max_depth=3)
    
    # 특정 메뉴 크롤링 예시
    # crawler.crawl_menu("https://www.skku.edu/skku/about/pr/greeting.do", max_depth=2)
    
    # 특정 페이지 크롤링 예시
    # crawler.crawl_specific_page("https://www.skku.edu/skku/about/pr/greeting.do") 
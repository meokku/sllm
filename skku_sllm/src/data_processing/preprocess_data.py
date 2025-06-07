#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime

class SKKUDataProcessor:
    """성균관대학교 크롤링 데이터 전처리 클래스"""
    
    def __init__(self, input_dir="data/raw/skku_data", output_dir="data/processed"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def clean_text(self, text):
        """텍스트 정제"""
        if not text or not isinstance(text, str):
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 여러 공백 문자를 하나로 대체
        text = re.sub(r'\s+', ' ', text)
        
        # 불필요한 특수문자 제거
        text = re.sub(r'[^\w\s\.,?!;:()\"\'%\-]', ' ', text)
        
        return text.strip()
    
    def load_jsonl_files(self):
        """JSONL 파일들을 로드하여 하나의 리스트로 통합"""
        all_data = []
        jsonl_files = glob.glob(os.path.join(self.input_dir, "*.jsonl"))
        
        print(f"JSONL 파일 {len(jsonl_files)}개를 처리합니다...")
        
        for jsonl_file in tqdm(jsonl_files):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        all_data.append(item)
                    except json.JSONDecodeError:
                        print(f"JSON 디코딩 오류: {line[:50]}...")
        
        print(f"총 {len(all_data)}개 항목을 로드했습니다.")
        return all_data
    
    def load_json_files(self):
        """JSON 파일들을 로드하여 하나의 리스트로 통합"""
        all_data = []
        json_files = glob.glob(os.path.join(self.input_dir, "*.json"))
        
        print(f"JSON 파일 {len(json_files)}개를 처리합니다...")
        
        for json_file in tqdm(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 데이터가 리스트인 경우 (여러 항목)
                    if isinstance(data, list):
                        all_data.extend(data)
                    # 데이터가 딕셔너리인 경우 (단일 항목)
                    else:
                        all_data.append(data)
            except Exception as e:
                print(f"파일 처리 오류: {json_file} - {e}")
        
        print(f"총 {len(all_data)}개 항목을 로드했습니다.")
        return all_data
    
    def remove_duplicates(self, data):
        """중복 데이터 제거"""
        # URL 기준으로 중복 제거
        url_seen = set()
        unique_data = []
        
        for item in data:
            # URL이 있는 항목인 경우
            if 'url' in item:
                url = item['url']
                if url not in url_seen:
                    url_seen.add(url)
                    unique_data.append(item)
            # URL이 없는 항목인 경우 (제목 + 내용 해시로 중복 체크)
            elif 'title' in item and ('content' in item or 'text' in item):
                content = item.get('content', '') or item.get('text', '')
                item_hash = hash(f"{item['title']}_{content[:100]}")
                if item_hash not in url_seen:
                    url_seen.add(item_hash)
                    unique_data.append(item)
            # 그 외 항목은 모두 포함
            else:
                unique_data.append(item)
        
        print(f"중복 제거 후 {len(unique_data)}개 항목이 남았습니다. ({len(data) - len(unique_data)}개 제거)")
        return unique_data
    
    def filter_by_content_length(self, data, min_length=50):
        """내용 길이가 너무 짧은 항목 필터링"""
        filtered_data = []
        
        for item in data:
            content = ""
            if 'content' in item:
                content = item['content']
            elif 'text' in item:
                content = item['text']
            
            if len(content) >= min_length:
                filtered_data.append(item)
        
        print(f"내용 길이 필터링 후 {len(filtered_data)}개 항목이 남았습니다. ({len(data) - len(filtered_data)}개 제거)")
        return filtered_data
    
    def standardize_format(self, data):
        """데이터 형식 표준화"""
        standardized_data = []
        
        for item in data:
            # 기본 필드 설정
            std_item = {
                "title": "",
                "content": "",
                "url": "",
                "date": "",
                "type": "webpage",
                "source": "skku.edu"
            }
            
            # 제목 필드
            if 'title' in item:
                std_item["title"] = self.clean_text(item['title'])
            
            # 내용 필드
            if 'content' in item:
                std_item["content"] = self.clean_text(item['content'])
            elif 'text' in item:
                std_item["content"] = self.clean_text(item['text'])
            
            # URL 필드
            if 'url' in item:
                std_item["url"] = item['url']
            elif 'link' in item:
                std_item["url"] = item['link']
            
            # 날짜 필드
            if 'date' in item:
                std_item["date"] = item['date']
            elif 'crawled_at' in item:
                std_item["date"] = item['crawled_at']
            
            # 유형 필드 (공지사항, 행사 등)
            if 'type' in item:
                std_item["type"] = item['type']
            elif 'is_notice' in item and item['is_notice']:
                std_item["type"] = "notice"
            
            # 제목이나 내용이 없는 항목은 제외
            if std_item["title"] or std_item["content"]:
                standardized_data.append(std_item)
        
        print(f"형식 표준화 후 {len(standardized_data)}개 항목이 남았습니다.")
        return standardized_data
    
    def convert_to_training_format(self, data):
        """학습 데이터 형식으로 변환
        
        일반적인 질의응답 형식으로 변환. 제목을 질문으로, 내용을 답변으로 설정.
        """
        training_data = []
        
        for item in data:
            # 제목과 내용이 모두 있는 경우만 처리
            if item["title"] and item["content"]:
                # 질문 생성 (제목 기반)
                question = item["title"]
                
                # 몇 가지 질문 형식 추가
                questions = [
                    question,  # 원래 제목
                    f"{question}에 대해 알려주세요",
                    f"{question}은(는) 무엇인가요?",
                    f"{question}에 대한 정보를 제공해주세요"
                ]
                
                # 답변 설정
                answer = item["content"]
                
                # 각 질문 형식별로 데이터 추가
                for q in questions:
                    training_data.append({
                        "instruction": "성균관대학교 학생들에게 도움이 되는 정보를 친절하게 제공해주세요.",
                        "input": q,
                        "output": answer,
                        "source_url": item["url"],
                        "source_type": item["type"]
                    })
        
        print(f"학습 데이터 형식으로 변환되었습니다. 총 {len(training_data)}개 생성")
        return training_data
    
    def process_all(self):
        """모든 전처리 과정 실행"""
        print("성균관대학교 크롤링 데이터 전처리를 시작합니다...")
        
        # 1. JSONL 파일 로드
        jsonl_data = self.load_jsonl_files()
        
        # 2. JSON 파일 로드
        json_data = self.load_json_files()
        
        # 3. 모든 데이터 통합
        all_data = jsonl_data + json_data
        print(f"총 {len(all_data)}개 항목을 처리합니다...")
        
        # 4. 중복 제거
        unique_data = self.remove_duplicates(all_data)
        
        # 5. 내용 길이 필터링
        filtered_data = self.filter_by_content_length(unique_data)
        
        # 6. 형식 표준화
        standardized_data = self.standardize_format(filtered_data)
        
        # 7. 학습 데이터 형식으로 변환
        training_data = self.convert_to_training_format(standardized_data)
        
        # 8. 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 원본 전처리 데이터 저장
        with open(os.path.join(self.output_dir, f"skku_processed_{timestamp}.json"), 'w', encoding='utf-8') as f:
            json.dump(standardized_data, f, ensure_ascii=False, indent=2)
        
        # 학습 데이터 형식 저장
        with open(os.path.join(self.output_dir, f"skku_training_{timestamp}.json"), 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"전처리가 완료되었습니다. 결과가 {self.output_dir} 디렉토리에 저장되었습니다.")
        print(f"- 표준화 데이터: {len(standardized_data)}개")
        print(f"- 학습 데이터: {len(training_data)}개")


# 실행 코드
if __name__ == "__main__":
    processor = SKKUDataProcessor()
    processor.process_all() 
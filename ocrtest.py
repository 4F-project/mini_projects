import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from keybert import KeyBERT
import re
import torch

# PaddleOCR 설정 (한국어 및 영어 지원)
ocr = PaddleOCR(lang='korean')

# 이미지 로드 및 전처리
image_path = 'fest.jpg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
contrast = cv2.convertScaleAbs(binary, alpha=1.5, beta=0)

# OCR로 텍스트 추출
result = ocr.ocr(contrast, cls=True)
words = [line[1][0] for line in result[0]]  # 단어 단위로 저장
print("ocr로 추출된 텍스트", words)

# 불필요한 기호 및 잘못된 인코딩 문자 제거
processed_words = [re.sub(r'[^\w\s가-힣A-Za-z]', '', word) for word in words]
processed_text = " ".join(processed_words)

# 다국어 지원 임베딩 모델 설정
model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

# 임베딩 생성 및 유사도 분석
embeddings = model.encode(processed_words, convert_to_tensor=True)
similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
threshold = 0.7
clusters = [set(processed_words[j] for j in range(len(processed_words)) if similarity_matrix[i][j] > threshold) for i in range(len(processed_words))]

# 전처리된 텍스트를 사용해 주제와 요약 생성
full_text = processed_text

# 1. 주제 분류 파이프라인 설정 (Zero-shot Classification)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["event", "announcement", "competition", "cultural foundation"]
classification = classifier(full_text, candidate_labels)
main_topic = classification['labels'][0]  # 주제 결정

# 2. 본문 요약 생성
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

# 3. 자동 키워드 추출 (KeyBERT 사용)
kw_model = KeyBERT(model='xlm-r-bert-base-nli-stsb-mean-tokens')
keywords = kw_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 2), stop_words='english')

# 결과 출력
print("주제:", main_topic)
print("요약:", summary)
print("자동 추출된 키워드:")
for kw in keywords:
    print(f"- {kw[0]} (중요도: {kw[1]:.2f})")













# C:/Program Files/Tesseract-OCR/tesseract.exe
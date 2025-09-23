## 이화여대 RAG 챗봇 Database

1. 학교 공식 홈페이지
2. 에브리타임, 이화이언 등 교내 커뮤니티
3. 뉴스/기사

## Structure
```
data/
  official/      # 학교 공식 홈페이지
  community/     # 에브리타임, 이화이언 정리
  news/          # 언론 기사, 보도자료
processed/       # 전처리된 데이터 (chunking 후 json 등)
```
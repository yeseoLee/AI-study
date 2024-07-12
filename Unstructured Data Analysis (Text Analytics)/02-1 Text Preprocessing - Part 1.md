# Introduction to NLP
## Natural Language Processing
- 음운론 / 통사론 / 구문 분석 / 의미 분석 / 원인 분석 단계를 거침
### Classical categorization of NLP
- Phonology (음운 분석)
	- Speech to Text (STT)
	- Speech Recognition
	- Text to Speech (TTS)
- Morphology (형태소 분석)
- Syntax (구조적인 관계 분석)
- Semantics (의미론적 분석)
- Pragmatics, Discourse (사람의 사회적인 작용과 연계되는 부분이라 현재 자연어처리 기술로는 구현이 어려움, rarely used)
### An example of NLP
- Lexical Analysis
- Syntax Analysis
- Semantic Analysis
- Pragmatic Analysis (함축적인 의미를 분석하기 어려움)
## Why is NLP hard?
### Programming Language vs Natural Language
- 프로그래밍 언어는 단순한 문법(기능적 어휘는 100단어 정도면 가능)
- 자연 언어는 이와 다름 (영어의 경우 10만개 이상)
	- 복잡한 문법
	- 모호성
	- 시간에 따른 단어의 변화 (생명체처럼 진화하기도 하고 사라지기도 함)
### Ambiguity of a natural language
- 중의적 표현 (ex. He saw the man with the telescope)
- 실제로는 더 복잡한 구문이 훨씬 많음
## Research Trends in NLP
- From rule-based approaches to **statistical approaches**
- From statistical approaches to **machine-learning(deep-learning)** approaches
  -> 연역적 방식에서 귀납적 방식으로의 자연어 처리 모델을 만드는 형태로 헤게모니의 변화
- **End-to-End Multi-Task** Learning (종단 학습)
	- 문서와 최종적인 output에 대한 label들만 주면 사람의 개입 없이 task를 수행할 수 있는 자연어 처리 모델 개발
- Performance Improvements with a huge model
- Statistical translation vs **deep learning-based translation**
## Data Quality in NLP
- ExoBrain Project
- Data Annotation as a Business Model
	- ScaleAI
	- BasicAI
	- Amazon SageMaker Ground Truth
		- Data Labeling Platform
	- DataMaker
	- 테스트웍스

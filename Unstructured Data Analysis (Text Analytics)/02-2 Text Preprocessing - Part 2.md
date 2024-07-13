# Lexical Analysis (어휘 분석)
### 어휘 분석의 목적
  일정한 순서가 있는 characters의 조합을 token으로 변경 (token: 의미를 가지는 character strings)
### 어휘 분석의 과정
- Tokenizing
- 품사 태깅 (POS tagging)
- NER(named entity recognition) 개체명 인식, 명사구 인식 등 수행
### 구조적 분석의 예시
- Part of speech
  각 토큰/단어에 대해 형태소 판별
- Named entity recognition (개체명 인식)
  e.g.) Obama: 사람, now: Date, ...
- Co-reference (문장 내에서 같은 것들을 지칭하는 것을 찾아주는 것)
  e.g.) She = Mrs.Clinton 
- Basic dependencies (의존 구문 분석)
  e.g.) 누가 무엇을 했느냐
## Lexical Analysis 1: Sentence Splitting
- 문장은 NLP에서 매우 중요하나, Topic Modeling과 같은 일부 Text Mining Task에서는  critical 하지 않음
- 문장 부호가 등장했을 때는 쉬우나, 마침표가 있지만 구분자가 아닐 때가 있어 규칙 기반으로만 문장을 분리하기는 쉽지 않다.
- 인용 부호(quotes)가 있어 문장이 중첩되는 경우도 있다.
## Lexical Analysis 2: Tokenization
- word tokens, number tokens, space tokens, ...
- 공백/구두점/숫자/특수문자 등을 tokenizer에 따라 제거하거나 제거하지 않는다. 즉 목적에 맞는 tokenizer를 사용하면 된다.
- 그러나 tokenization 또한 쉽지 않다(완벽하지 않다).
## Lexical Analysis 3: Morphological Analysis
### Morphological Variants: Stemming and Lemmatization
- **Stemming**: 단어의 (사전에 등장하지 않을 수 있는)base form을 찾음
	- 정보 추출 관점에서 많이 사용
	- rule-based 알고리즘으로 구현 (suffix-stripping 기반)
	- 대표적으로 영어 관점에서 Porter stemmer가 있음
	- 단순하고 빠르다는 장점
	- 규칙이 언어 종속적이고, 생성한 단어가 언어에 존재하지 않을 수 있고, 서로 다른 단어가 같은 stem으로 stemming 될 수 있는 단점이 있음
- **Lemmatization**: lemma(품사를 보존하는 단어의 원형)을 찾음
	- 실질적으로 존재하는 단어인 lemma(root form)을 찾음
	- stemming에 비해 적은 오차
	- 좀 더 복잡하고 느림
	- Semantics(의미 분석)이 중요한 상황에서 사용
- e.g.) 
  ![[stemming_lemmatization.png]]
## Lexical Analysis 4: Part-of-Speech (POS) Tagging
- 문장이 주어졌을 때, 각각의 토큰들의 POS tag를 예측
- Input: tokens / Output: most appropriate tag
- 같은 token이어도 품사에 따라 POS가 달라짐 (상황, 문맥 파악)
### statistical POS Tagging
- 토큰을 하나씩 스캔하여 형태소를 할당
- 단어의 순서에 해당하는 확률이 어떻게 될것이냐를 통해 적절한 형태소를 할당
- manually tagged corpus가 필요 (언어학적인 지식을 가진 전문가가 정답을 달아주어야 함)
- 주의: parsing: 문장 구조를 찾아내는 것 / POS: 품사를 찾아내는 것
- Tagsets
### POS Tagging Algorithms
- Training: manually annotated corpus 필요
- Tagging algorithm: 주어진 텍스트에 대해 학습 데이터를 통해 만든 모델(learned parameters)를 적용시켜 적절한 품사를 지정
- 학습시 사용한 corpus와 같은 도메인에서는 좋은 성능을 보임
- 이전에는 Decision Trees, Hidden Markov Models, Support Vector Machines 등을 주로 사용
- 최근에는 트랜스포머 기반의 pre-trained 모델 (BERT 등)을 변형하여 학습에 사용
### POS Tagging Algorithms 예시
#### Pointwise prediction: 각각의 단어들을 개별적으로 classifier을 통해 예측 (e.g. **Maximum Entropy Model**, SVM)
- 일정한 sequence를 탐색하고 단어를 하나씩 양쪽 Context에 해당하는 단어들을 보며 shifting하며 예측
- Encode features for tag prediction
- Tagging Model
#### Probabilistic models
문장이 주어졌을 때, 가장 그럴듯한 tag sequence를 찾는다. -> argmax $P(Y|X)$
- **Generative sequence models**: w1 -> w2 -> w3 -> ... 순차적으로 할당
	- 베이즈 규칙을 통해 확률 분해 -> argmax $P(X|Y)P(Y)$
	- $P(X|Y)$: 단어와 품사의 관계 (e.g. natural은 아마 JJ일 것) / $P(Y)$: 품사와 품사의 관계 (e.g. 관사 이후 관사가 바로 나오는 경우는 거의 없음)
- **Discriminative sequence models**: (w1, w2, w3, ...) 각각의 단어들에 대한 토큰을 일괄적으로 한번에 예측
	- Conditional Random Field (CRF)
####  Neural network-based models (e.g. BERT, ...)
- Window-based vs sentence-based
	- Window-based: 전체 단어 시퀀스에 대해 일부분(윈도우)을 보고 태그를 찾는 방식
	- sentence-based: 하나의 neural network 입력이 whole sentence임
- RNN, LSTM, GRU 구조 등을 많이 시도하였음
- Hybrid model: LSTM(RNN) + ConvNET + CRF

## Lexical Analysis 5: Named Entity Recognition
NER: 각각의 elements가 pre-defined categories로 구분
### Approaches 
- Dictionary/Rule-based
	- List-lookup: 단순하고 빠르지만 리스트가 변하거나 업데이트가 된 것을 관리하는 것은 쉽지 않음
	- Shallow Parsing (증거)
		- Location: Cap Word + {Street, Boulevard, Avenue}
		- e.g: Wall Street
- Model-based
	- MITIE
	- CRF++
	- CNN

## BERT for Multi NLP Tasks
- 하나의 NLP task를 위한 모델을 만드는게 아닌 multi task를 처리하는 구조를 만드는 것이 트렌드
- Google Transformer 기반
- http://jalammar.github.io/illustrated-transformer/ (**이해될 때 까지 반복해서 볼것!**)
- Classification / QA, POS Tagging 등의 tasks에 사용 

# 요약
- 어휘 분석(Lexical Analysis)에 대한 이해: 입력되는 문자들을 입력되는 순서대로 모아서 단어를 구성하고 어떤 종류의 단어인가를 구별하는 일  
- 문장 분리 - 토큰화 - 형태론적 분석 (어간추출, 표제어 추출) - 품사 태깅(통계적 모델, 확률적 모델, 신경망 기반 모델) - 개체명 인식(규칙 기반, 모델 기반)
- 최근 NLP는 하나의 task를 위한 모델을 만드는게 아닌 multi task를 처리하는 구조를 만드는 것이 트렌드 (e.g. BERT)
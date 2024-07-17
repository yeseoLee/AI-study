# Bag of Words
어떻게 가변 길이의 문서를 고정 길이의 숫자형 벡터로 변환할 것인가?
## Bag of Words: Motivation
### Document Representation
- 어떻게 하나의 Document를 구조화된 vector/matrix 형태(Vector Space ModeL)로 변환할 것인가 (Transform unstructured data into structured data)  
## Bag of Words:Idea
- 가정: 문서들은 **순서를 무시하는 단어들의 집합체**이다.
- 단어 하나 하나를 atomic symbol로 고려하여 discrete space로 표현 
### Term-Document Matrix
TDM vs DTM(각각의 객체가 행이고, 변수들이 열일 때)
- Binary representation: 해당하는 단어의 등장 여부
- Frequency representation: 해당하는 단어의 등장 빈도
### Bag of words Representation in a Vector Space
- 가정: 컨텐츠는 단어의 사용 빈도로부터 추론될 수 있다.
- **단어의 순서를 고려하지 않기 때문에** 의미가 다른 두 문장이 동일하게 표현될 수 있다. (e.g. A is better than B /  B is better than A 가 BOW 표현에서는 일치한다.)
- **reconstruct 불가** (tdm에서 original text로 복구할 수 없다.)
## Stop Words
- 해당하는 문서를 이해하는데 필요없는 (주요 정보를 가지지 않고 문법적인 역할만 수행하는) 단어
- 알고리즘으로 제거할 수도 있으나 불용어 리스트가 미리 존재하고 이를 매칭하여 제거한다.  
- e.g.) SMART Stop Words, MySQL Stop Words
# Word Weighting
## Term-Frequency (TF)
${tf}_{(t,d)}$ : **특정 document d에서 특정 term t가 얼마나 등장했는가**
## Document Frequency (DF)
${df}_{t}$ : **특정 term t가 몇 개의 document에 등장했는가**
### Issues on DF
- 자주 등장하는 term에 비해 희박하게 등장하는 term이 특정 문서에 대해 중요도가 높다 (e.g. is, can, the, of, ...는 DF가 높으나 중요하지 않을 가능성이 높다)
- Common Term 보다 Rare Term에 더 높은 가중치를 주어야 한다. -> IDF
## Inverse Document Frequency (IDF)
${idf}_{t} = {log}_{10}(N/df_t)$ (기하급수적으로 늘어나기 때문에 주로 로그를 씌워 사용한다)  
## TF-IDF
$$ TF-IDF(w) = tf(w) \times log(\frac{N}{df(w)}) $$ 
- Term Frequency는 커야하고, Inverse Document Frequency는 작아야 한다.
- 단어는 하나의 **target 문서에 자주 등장할수록**, 전체 corpus 중에서 **소수의 문서에만 등장할 수록** 중요도가 높다.
### |V|-dimensional vector space
- Terms are axes of the space (단어가 공간의 축)
- Documents are points or vectors in this space (문서가 공간의 점 또는 벡터)
- **Very high dimensional**: need to reduce the number of features (고차원)
- **Sparseness**: most entries are zero (대부분이 0인 희소한 벡터 표현)
## TF-IDF Variants
### Most commonly used TF-IDF Variants in general
#### TF Variants
- l(logarithm): $1+log(tf_{t,d})$
#### DF & IDF Variants
- t(idf): $log(\frac{N}{df_t})$
#### Normalization
- c (cosine): $\frac{1}{\sqrt{w_1^2+w_2^2+\dots+w_M^2}}$
### Effects of TF-IDF Variants
#### Comparative Study
- TF-IDF Variants 조합에 따라 성능이 다름
- 통계적으로 가장 좋은가는 아님
- 그러나 조합에 따라 상당히 큰 성능 차이가 나타남
- => Documents를 어떻게 representation하느냐에 따라 성능 차이가 나타난다.
# N-Grams
### N-Gram-based Language Models in NLP
- Use the previous N-I words in a sequence to predict the next word
### N-Gram in Text Mining
- Some phrses are very useful in text clustering/categorization!
	- Six sigma, big data, etc. (phrase가 하나의 관용구로 쓰일 때는 단어 각각으로 보는 것이 아닌 N-Gram 기준으로 보는게 더 타당할 것 / big data는 각각의 뜻을 이어서 보기보다 big data 하나로 보는 것이 더 타당함)
- Term-frequency for n-grams can be utilized
- Domain-dependent
### Empirical evaluation
- Data sets: 20 newsgroup data set + 21578 REUTERS newswire articles
- Classification algorithm: RIPPER

**Unigram에 비해 N-Gram을 사용했을 때 성능이 올라가는 정도가 찾는데 드는 비용에 비해 만족할만한가에 대한 의문점이 있음**

# 요약
- Bag Of Words
	- 가변 길이의 문서를 고정 길이의 숫자형 벡터로 변환
	- 각 단어를 고유하고 원자적인 기호로 고려하여 개별 공간으로 표현 
	- 순서를 고려하지 않는다와 복원이 불가하다는 단점
- Word Weighting
	- 단어의 중요도에 따라 가중치를 부여
	- TF-IDF: TF와 IDF의 곱. 하나의 **target 문서에 자주 등장할수록**, 전체 corpus 중에서 **소수의 문서에만 등장할 수록** 중요도가 높다
- N-Grams
	- 단어를 개별적으로을 보는 것이 아닌 n개씩 보는 것이 합리적인 경우
	- 성능이 올라가는 정도가 N-Gram을 찾는데 드는 비용에 비해 만족할만한가에 대한 의문점이 있음


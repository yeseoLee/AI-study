# Word-level: GloVe
### Limitations of Word2Vec
- 과도하게 사용된 단어에 대해 너무 많이 학습 
- e.g.) 관사 the가 나올 확률이 너무 높음 P(w|the)
## Glove
- matrix factorization method 기반
- V x V 크기의 큰 행렬
- $X_{ij}$ = i가 j와 함께 등장하는 빈도,  $X_i$ = 전체 코퍼스에서 단어 i가 등장한 횟수
- $P_{ij} = P(j|i) = \frac{X_{ij}}{X_i}$ 
### Motivation
- 특정 k라는 단어가 ice와 연관이 높고 steam과는 아니라면 $P_{ik} / P_{jk}$ 가 커야함  
- 단어 k가 steam과 관련성이 높지만 ice와는 아니라면 $P_{ik} / P_{jk}$ 는 작아야함  
- 단어 k가 둘 다 관련있다면 비율은 1에 가까워져야함
### Formulation
- 함수 F를 이용하여 세 단어들의 관계를 표현
  $$F(w_i,w_j,\tilde{w}_k) = \frac{P_{ik}}{P_{jk}}$$
- $w_i$와 $w_j$의 관계를 substraction(뺄셈)으로 표현
  $$F(w_i - w_j,\tilde{w}_k) = \frac{P_{ik}}{P_{jk}}$$
- 두 단어 사이의 차이(관계)$w_i - w_j$ 와 context word $\tilde{w}_k$ 사이의 링크를 만들어 주기 위해 내적으로 하나의 스칼라 값을 만들어줌
  $$F \left((w_i - w_j)^T \tilde{w}_k \right) = \frac{P_{ik}}{P_{jk}}$$
  ### Homomorphism
  아래 수식 대신, 아까 변형한 수식을 이용하여 표현한다.  
$$\frac{P(solid|ice)}{P(solid|steam)}$$  
$$F \left((w_i - w_j)^T \tilde{w}_k \right) = \frac{P_{ik}}{P_{jk}}$$  
그렇다면 다음과 같이 표현할 수 있다.  
$$\frac{P(solid|ice)}{P(solid|steam)} = F \left((ice - stream)^T solid \right)$$  
ice와 steam의 위치를 바꾸면 아래와 같이 빼기 순서가 달라진다.  
$$\frac{P(solid|steam)}{P(solid|ice)} = F \left((steam - ice)^T solid \right)$$  
두 식을 정리하면 다음과 같다. (즉, ice와 steam의 위치를 바꾸면 역수가 된다.)  
$$F \left((ice - steam)^T solid \right) = \frac{P(solid|ice)}{P(solid|steam)} = \frac{1}{F \left((steam - ice)^T solid \right)}$$  

`Input`에 대한 항등원 관계를 정리  
$$(ice - steam)^T solid = - (steam - ice)^T solid $$  
`F의 output`에 대한 항등원 관계를 정리  
$$F \left((ice - steam)^T solid \right) = \frac{1}{F \left((steam - ice)^T solid \right)}$$  

- Need a homomorphism **from $(R,+)$ to $(R_{>0},\times)$**
   **= 입력을 덧셈의 항등원으로 바꿔주면 함수의 출력값은 곱셉의 항등원으로 나오게 되는 Mapping이 필요**하다.
   **= 입력에서의 덧셈은 함수값에서의 곱셈으로 빠져나올 수 있어야 한다.**
- f(a+b) = f(a)f(b)를 만족해야 하므로, 그 중 우리가 아는 가장 쉬운 함수인 **F(x) = exp(x)**
### Solution

위 식에서 F(x) 대신 exp(x)로 정리하고 로그를 씌우면
$$w_i^T \tilde{w}_k = logP_{ik} = logX_{ik} - logX_i$$
$logX_i$ 를 $b_i+\tilde{b_k}$ 로 표현하여 정리하면 
$$w_i^T \tilde{w}_k + b_i+\tilde{b_k} = logX_{ik}$$

### Objective Function
 $$ J = \sum^V_{i,j=1}f(X_{ij})\left(w_i^T \tilde{w}_k + b_i+\tilde{b_k} \right)^2$$
$f(X_{ij})$ 
- 고 빈도 단어들의 학습에 대한 가중치를 낮춰주는 역할
- $(x / x_{max})^a$ if $x < x_{max}$ otherwise $1$

---
# Word-level: FastText
### Limitations of NNLM, Word2Vec, and GloVe
- 단어가 가지는 morphology 특성을 무시하고 있음
- morphologically rich languages에 대해 적용하기 어렵다 (터키어, 필란드어 등)
### Goal
- character n-grams 표현을 학습
- 단어의 분산 표상은 n-gram vectors의 총합으로 표현

### Revisit Negative Sampling in Word2Vec 
- Word2Vec의 Score는 두 임베딩 사이의 내적을 통해 계산
- FastText의 Score는 w에 대한 n-grams를 정의한 다음, 벡터 표현을 전부다 더해서 내적

$$score(w,c) = \sum_{g \in g_w}z_{g}^Tv_c$$
- e.g.) apple 하나만 embedding하는 것이 아닌 a부터 ap, app ... apple 전부 더한 것이 apple의 embedding이다. 라는 개념
### subword model
- n-gram representation

### Word Embedding Examples
- Word Embedding with two different langauges
  한 언어만 하는 것이 아닌 서로 다른 언어들과 임베딩
- Word Embedding with Images
  단어만 임베딩하는 것이 아닌 이미지, 영상 등의 멀티모달 데이터를 통해서도 임베딩 

---
# 요약
- GloVe
	- Word2Vec는 고빈도 단어를 과도하게 학습하는 문제를 해결하고자 등장
	- F(x)
		- 입력을 덧셈의 항등원으로 바꿔주면 함수의 출력값은 곱셉의 항등원으로 나오게 되는 Mapping이 필요
		- 입력에서의 덧셈은 함수값에서의 곱셈으로 빠져나올 수 있어야 함
		- F(x) = exp(x)
	- 목적 함수
		- $f(X_{ij})$ 는 고 빈도 단어들의 학습에 대한 가중치를 낮춰주는 역할
- FastText
	- 기존 임베딩들이 단어가 가지는 morphology 특성을 무시하는 문제를 해결하고자 등장
	- character n-grams 표현을 학습
	- 단어의 분산 표상은 n-gram vectors의 총합으로 표현
	- subword model

> [07-3 Topic Modeling Part 3 - LDA Inference](https://youtu.be/iwMSCsiL6wQ)

# LDA Inference
- 문서가 주어졌을 때 latent variables의 사후확률은 다음과 같다.

$$p(\phi, \theta, Z | W) = \frac{p(\phi, \theta, Z, W)}{\int_{\phi} \int_{\theta} \sum_Z p(\phi, \theta, Z, W)}$$
- <font color='red'>사후 확률을 계산하는 것은 불가능하다.</font> (분모를 계산할 수 없어서 근사해야한다.)
- 사후 확률 추정 알고리즘
	- Mean field variational methods
	- Expectation propagation
	- <font color='blue'>Collapsed Gibbs sampling</font> -> 강의에서는 가장 직관적인 이 방식으로 설명하겠다.
	- Collapsed variational inference
	- Online variational inference

## LDA: Dirichlet Distribution
- 왜 Dirichlet 분포일까?
### Binomial & multinomial
- <font color='blue'>Binomial distribution (이항 분포)</font>
	- 어떤 사건들의 결과물이 두가지일 때 성공이 몇번이나 발생할 것인가 
	- (n: 시행횟수, p: 성공확률, x: 성공횟수)    
  $$p(X=x|n,p) = \binom{n}{x} p^x (1-p)^{n-x}$$
- <font color='blue'>Multinomial distribution (다항 분포)</font>
	- k개의 가능한 결과물에 대해서 $x_1$부터 $x_k$까지 발생할 확률을 어떻게 계산할 것인가
  $$p(x_1, ..., x_k|n, p_1, ..., p_k) = \frac{N!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i}, \quad \sum_{i} x_i = N, \ x_i \geq 0$$

### Beta distribution
  $$p(p|\alpha, \beta) = \frac{1}{B(\alpha, \beta)} p^{\alpha-1} (1-p)^{\beta-1}$$
- p가 0과 1 사이의 값을 가질 때, p를 이항 분포의 매개변수로 생각하면 베타를 '분포들의 분포'(이항 분포들의 분포)라고 생각할 수 있다.
  $$B(\alpha, \beta) = \int_0^1 t^{\alpha-1}(1-t)^{\beta-1} dt \qquad B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$$
$$\Gamma(x) = \int_0^\infty \frac{t^{x-1}}{exp(t)} dt, \quad (x > 0) \qquad \Gamma(n) = (n-1)!$$
### Dirichlet distribution
 $$p(P = \{p_i\}|\alpha_i) = \frac{\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)} \prod_i p_i^{\alpha_i - 1}$$
 - 결국 베타 분포의 일반화된 확장판이다. 
   ($\frac{1}{B(\alpha, \beta)}$를 감마로 표현했을 때의 수식과 비교해보면 알 수 있다.)
	- 베타 분포는 binomials에 해당하는 성공 확률의 분포이다.
	- Dirichlet은 Multinomials의 성공 확률들에 대한 확률 분포이다.
- Dirichlet is conjugate prior of multinomial
  - conjugate prior distribution (켤레 사전 분포): 사후확률을 계산함에 있어 사후 확률이 사전 확률 분포와 같은 분포 계열에 속하는 경우
### Important properties of Dirichlet distribution
#### Posterior is also Dirichlet
$$p(P = \{p_i\}|\alpha_i) = \frac{\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)} \prod_i p_i^{\alpha_i - 1}$$

$$p(x_1, ..., x_k | n, p_1, ..., p_k) = \frac{n!}{\prod_i^k x_i!} \prod_i^k p_i^{x_i}$$

$$p(\{p_i\}|x_1, ..., x_k) = \frac{\Gamma(N + \sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i + x_i)} \prod_i p_i^{\alpha_i + x_i - 1}$$

- 우리가 현재 관측했던 정보들을 집어넣었을 때의 확률값이 여전히 Dirichlet 분포이다.
- 사후 분포가 여전히 Dirichlet이라는 것은 Dirichlet 분포를 정의하고, 각각의 범주에 대한 관측치를 Count하여 확률을 업데이트했을 때, 그것 역시 Dirichlet에 의해 정해진다는 뜻이다.

#### The Parameter $\alpha$ controls the mean shape and sparsity of $\theta$
- 파라미터 알파가 평균의 형태와 세타의 희소성을 조정한다.
- A Dirichlet with $\alpha_i < 1$ favors extreme distribution 
  (알파가 1보다 작으면 극단적인 확률분포가 발생한다.)
#### Sampling results from Dirichlet with diffrent $\alpha$
- 알파가 커질수록 개별 문서에서 모든 토픽들이 균등하게 분포할 것이라는 가정
- 알파가 작아질수록 개별 문서에서 특정 토픽들이 집중적으로 나타날 것이라는 가정
- 알파와 베타는 최적화 대상이 아니며, 하이퍼파라미터이다. (세타, 파이, 제트를 찾는 것임)
- (이미지는 강의노트를 참고)

## LDA Inference
### Posterior distribution
$$p(z, \phi, \theta_w, \alpha, \beta)$$
$$p(\phi_{1:K}, \theta_{1:D}, z_{1:D}, w_{1:D}) = \prod_{i=1}^K p(\phi_i|\beta) \prod_{d=1}^D p(\theta_d|\alpha) \left(\prod_{n=1}^N p(z_{d,n}|\theta_d) p(w_{d,n}|\phi_{1:K}, z_{d,n})\right)$$
- 결합확률분포가 최대가 되게하는 파이와 세타와 제트를 찾자 (알파와 베타는 하이퍼파라미터) 

### Gibbs Sampling
- $z_k$를 제외한 나머지 상태들은 다 주어져있다는 가정 하에 $z_k$를 샘플링. k를 바꿔가면서 계속 샘플링한다.
-  $z_{-d,n}$ 는 단어의 assignments 중에서 $z_{d,n}$를 제외한 나머지를 표현하는 것
#### Monte Carlo method
- https://en.wikipedia.org/wiki/Monte_Carlo_method
#### Markov Chain Monte Carlo sampling
- 몬테카를로 방법에 마르코프 체인(과거에 일어난 사건들의 확률을 통해 미래를 예측하는 기법)을 추가

#### Gibbs Variants
- Gibbs Sampling
	- a 고정 b,c 샘플링
	- b 고정 a,c 샘플링
	- c 고정 a,b 샘플링
- Block Gibbs Sampling
	- a,b 고정 c 샘플링
	- c 고정 a,b 샘플링
- <font color='blue'>Collapsed Gibbs Sampling</font>
	- a 고정 c 샘플링
	- c 고정 a 샘플링
	- **이 과정을 반복할 때 b 과정은 사라지면서 굳이 샘플링할 필요 없다.**

## LDA Inference: Collapsed Gibbs Sampling
### $z_i$에 대한 조건부 사후 분포는 다음과 같이 주어짐
$$\begin{align}
p(z_i = j | Z_{-i}, W) &\propto p(z_i = j, Z_{-i}, W) \\
&= p(w_i | z_i = j, Z_{-i}, W_{-i}) \, p(z_i = j | Z_{-i}, W_{-i}) \\
&= p(w_i | z_i = j, Z_{-i}, W_{-i}) \, p(z_i = j | Z_{-i})
\end{align}$$
- $p(z_i = j | Z_{-i}, W)$: i번째 단어를 제외하고 나머지는 topic이 할당이 되어있을 때 i번째 단어의 토픽이 j번째 토픽으로부터 나왔을 확률
- **첫 번째 항이 likelihood이고 두 번째 항이 prior처럼 작용한다.**
- # **TODO: 43분 ~ 54분 다시 듣고 정리** 
### Gibbs Sampling Equation
$$p(z_i = j | Z_{-i}, W) \propto \frac{n_{-i,j}^{(w_i)} + \beta}{n_{-i,j}^{(\cdot)} + V\beta} \times \frac{n_{-i,j}^{(d)} + \alpha}{n_{-i,j}^{(d)} + K\alpha}$$
- i번째 단어는 j번째 토픽으로부터 왔을 확률은 두가지의 확률(posterior, prior probability)에 비례한다.
- Need to record four count variables
	- Document-topic count $n_{-i,j}^{(d)}$
	  현재 doc에 $w_i$제외하고 Topic j에 할당된 단어의 수
	- Document-topic sum $n_{-i,j}^{(d)}$
	  doc에 $w_i$ 제외한 전체 단어의 수
	- Topic-term count $n_{-i,j}^{(w_i)}$
	  $w_i$ 제외하고 Topic j에 할당된 동일한 w의 수
	- Topic-term sum $n_{-i,j}^{(\cdot)}$
	  $w_i$ 제외하고 j번째 Topic에 할당된 전체 단어 수

### Parameter Estimation
$$\phi_{j,w} = \frac{n_{jw}^{(j)} + \beta}{\sum_{w=1}^V n_{jw}^{(j)} + V\beta} \qquad \theta_j = \frac{n_j^{(d)} + \alpha}{\sum_{z=1}^K n_z^{(d)} + K\alpha}$$
- $\phi_{j,w}$ : j번째 토픽에 w가 가지고 있는 비중
	- $n_{jw}^{(j)}$ : 실제 w가 j번째 토픽에 할당된 횟수
	- $\sum_{w=1}^V n_{jw}^{(j)}$ : j번째 토픽에 할당된 전체 단어의 수
	- $n_{jw}^{(j)}$를 $\sum_{w=1}^V n_{jw}^{(j)}$로 나눠줄 때 베타라는 디리클레 분포의 하이퍼파라미터를 통해 스무딩
	- 베타가 클수록 uniform한 분포를 갖는다.
- $\theta_j$ : d번째 단어에 j번째 토픽이 갖는 비중
	- $n_j^{(d)}$ : 문서 d 중에서 j번째 토픽에 할당된 단어의 수
	- $\sum_{z=1}^K n_z^{(d)}$ : 문서 d의 모든 토픽에 대한 단어의 수 (문서 d의 전체 단어 수)
	- 알파도 동일하게 스무딩 벡터 역할
	- 알파가 클수록 uniform한 분포를 갖는다.

- # **TODO: 이하 다시 듣고 정리** 
### Gibbs Sampling Equation: Another Form
## LDA Inference: Collapsed Gibbs Sampling
### Collapsed Gibbs Sampling
### Randomly assign topics
### Sampling
### What is the conditional distribution for this topic
#### Part 1: How much does this document like each topic?
- 현재 문서가 특정 토픽을 얼마나 선호하는가?
#### Part 2: How much does each topic like the word?
- 현재 토픽이 특정 단어를 얼마나 선호하는가?
#### Geometric interpretation

# LDA Evaluation
## LDA Evaluation & Model Selection
### How many topics are optimal?
- Log-likelihood for Gibbs Sampling 
- Perplexity
- Topic weights is determinated for the new data (holdout dataset) using Gibbs sampling
- Term distributions for topics are kept fixed from the training corpus
**정량적 지표로 Perplexity가 많이 쓰이나 토픽 모델링 특성상 오히려 정성적 분석이 더 최적일 수 있음**
### Model Selection based on Perplexity
### LDA Visualization

---
# 요약

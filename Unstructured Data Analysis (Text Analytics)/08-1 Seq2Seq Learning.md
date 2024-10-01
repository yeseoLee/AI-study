>[08-1 Seq2Seq Learning](https://youtu.be/0lgWzluKq1k?si=nk8Sl9Jp7XQsz-Zy)
# Sequence to Sequence (Seq2seq) Learning
- [Jay Alamar - Seq2seq Models With Attention 시각화](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
## Sequence-to-sequence model
- [sutskever et al. 2014](https://arxiv.org/pdf/1409.3215)
- [cho et al 2014](https://arxiv.org/pdf/1406.1078)
- 입력으로 아이템들(단어, 문자, 이미지 프레임 등등...)의 시퀀스를 받는다.
- 출력으로 아이템들의 또다른 시퀀스를 반환한다.
## Encoder-Decoder
- 인코더는 인풋 시퀀스의 각 아이템을 처리하고 정보를 컴파일하여 하나의 벡터(context vector)로 생성
- 인코더는 context vector를 디코더에 넘겨주고, 디코더는 item-by-item으로 아웃풋을 생성
- 가장 고전적인 방식으로는 RNN으로 구현
### RNN 방식 Encoder-Decoder
- 각각의 입력이 들어갈 때마다 hidden state가 한번씩 업데이트
- 최종적으로 모든 입력이 들어왔을 때 **가장 마지막 hidden state가 context vector가 됨**
- 인코더가 context vector를 디코더에 전달하고 디코더는 순차적으로 아웃풋을 생성

## Attention
- Vanilla RNN 구조에서는 가장 뒤쪽 아이템에 영향을 많이 받고, 앞쪽 아이템에는 영향을 거의 받지 못함
	- 그래서 LSTM이나 GRU를 만들었으나 Long-Term Dependency를 완화하는 수준이지 해결해주지는 못했음
- Seq2Seq 모델에서 Context Vector 자체가 긴 시퀀스를 처리하는데 어려움이 있음 (Bottleneck Problem)
- Attention은 모델이 각각의 인풋 시퀀스 중에서 현재 아이템이 주목해야할 부분에 Connection(혹은 가중치)를 주어 해당 파트의 정보를 잘 활용할 수 있도록 하는 구조
### Bahadanau attention
[Bahdanau et al. 2014](https://arxiv.org/pdf/1409.0473)  
- **Attention Score 자체를 학습하는 Neural Network 모델이 존재**
### Luong attention
[Luong et al. 2015](https://arxiv.org/pdf/1508.04025)  
- **Attention Score를 따로 학습하지 않고** 현재 hidden state와 과거의 hidden state 간의 유사도를 측정하여 Attention Score를 만듦
- 두 어텐션 매커니즘 사이의 성능이 크게 차이가 나지 않아 Luong attention이 더 실용적으로 사용됨

## Attention in Seq2Seq Learning
- 인코더가 디코더에 더 많은 정보를 넘겨주게 됨
	- 가장 마지막 hidden state만을 넘겨주는 것이 아닌 모든 hidden state를 넘겨주게 됨
### Attention model vs classic Seq2Seq model
어텐션 디코더는 인코더가 전달하는 여러 hidden state를 가지고 추가적인 단계를 거침  
- 인코더가 전달한 hidden state를 전부 확인 - 각각의 hidden state들은 해당하는 단어의 sequence에 가장 영향을 많이 받는다는 가정 (e.g. hidden state 2는 두번 째 입력 단어의 정보를 가장 많이 보존하고 있을 것)
- 각 hidden state들에 대한 score를 제공 
  (여러 방법이 있지만 가장 간단한 방법은 decoder hidden state와 encoder hidden state간의 내적)
- softmax를 수행하여 해당하는 값들을 전부 결합하여 하나의 weighted vector를 만듦(스코어 값이 클 수록 현재 decoding 단계에서 중요한 정보)
- 그렇게 나온 weighted vector를 context vector로 사용
- hidden state vector와 context vector를 concat하여 사용 
- feed-forward neural network를 통해 단어를 출력 & 다음 단계 입력으로 집어넣음

---
# 요약

## 비트코인 가격 예측

모델로 LSTM을 사용하였습니다.

bit_train.py 는 데이터를 학습시키는 용도이며, bit_test.py는 학습된 모델을 사용하여 upbit API를 통한 코인가격을 예측합니다.


### 환경
- python 3.6
- pandas 
- numpy
- tensorflow 1.5.0(1.12.0 에서도 작동 확인)


### 사용법
- training : 
python bit_lstm_train.py --save_path 저장위치
- testing :
python bit_lstm_test.py --save_path 저장위치

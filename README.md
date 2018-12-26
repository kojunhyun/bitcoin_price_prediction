## 비트코인 가격 예측

모델은 LSTM 모델을 사용하였습니다.

환경 :
- python 3.6
- pandas 
- numpy
- tensorflow 1.5.0(1.12.0 에서도 작동 확인)


사용법 :
- training : 
python --save_path model/ --train True
- testing :
python --save_path model/ --train False


# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
import os

def data_reader():
    # 데이터를 로딩한다.
    stock_file_name = 'bitcoin_usdt_1hours.csv' # 데이터 파일
    encoding = 'euc-kr' # 문자 인코딩

    #판다스이용 csv파일 로딩 header가 없을 경우
    #names = ['opening_date(kor)','closing_date(kor)','opening_price(dollor)','closing_price(dollor)','high_price(dollor)','low_price(dollor)','trade_price(dollor)','trade_volume'] 
    #raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding)

    #판다스이용 csv파일 로딩 header 가 있을경우
    raw_dataframe = pd.read_csv(stock_file_name, header=0, encoding=encoding)

    raw_dataframe.info() # 데이터 정보 출력
    return raw_dataframe

def min_max_scaling(x, d_min, d_max):
    re_x = x.values.astype(np.float) # 금액,거래량,지표데이터의 문자열을 부동소수점형으로 변환한다
    
    data_min = np.array(d_min)
    data_max = np.array(d_max)

    data_min = data_min.reshape(len(data_min), 1)
    data_max = data_max.reshape(len(data_max), 1)
    
    reg_data = (re_x - data_min) / (data_max - data_min + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원  

    return reg_data, data_min, data_max


def fnMA(m_DF):
    # 이동평균
    m_DF['MA5'] = m_DF['closing_price(dollor)'].rolling(window=5).mean()
    m_DF['MA20'] = m_DF['closing_price(dollor)'].rolling(window=20).mean()
    m_DF['MA60'] = m_DF['closing_price(dollor)'].rolling(window=60).mean()
    m_DF['MA120'] = m_DF['closing_price(dollor)'].rolling(window=120).mean()
    return m_DF


def fnStoch(m_Df, n=10): # price: 종가(시간 오름차순), n: 기간
    sz = len(m_Df['closing_price(dollor)'])
    if sz < n:
        # show error message
        raise SystemExit('입력값이 기간보다 작음')
    tempSto_K=[]
    for i in range(sz):
        if i >= n-1:
            tempUp =m_Df['closing_price(dollor)'][i] - min(m_Df['low_price(dollor)'][i-n+1:i+1])
            tempDown = max(m_Df['high_price(dollor)'][i-n+1:i+1]) -  min(m_Df['low_price(dollor)'][i-n+1:i+1])
            tempSto_K.append( tempUp / tempDown )
        else:
            tempSto_K.append(0) #n보다 작은 초기값은 0 설정

    m_Df['Sto_K'] = pd.Series(tempSto_K,  index=m_Df.index)    
    m_Df['Sto_D'] = pd.Series(m_Df['Sto_K'].rolling(window=6, center=False).mean())
    m_Df['Sto_SlowD'] = pd.Series(m_Df['Sto_D'].rolling(window=6, center=False).mean())
    return m_Df


def fnMACD(m_Df, m_NumFast=12, m_NumSlow=26, m_NumSignal=9):
    # MACD
    """
    MACD = EMA(numFast) - EMA(numSlow)
    EMA(Exponential Moving Average ; 지수이동평균)
    numFast: 기간
    numSlow: 기간
    numSignal: 기간
    MACD(numFast, numSlow, numSignal)
    numFast, numSlow 기간으로 MACD값을 산출하고 이를 c기간의 EMA를 시그널 선으로 활용합니다.
    일반적으로 MACD(12, 26, 9)가 많이 사용됩니다.
    """
    m_Df['EMAFast'] = m_Df['closing_price(dollor)'].ewm( span = m_NumFast, min_periods = m_NumFast - 1).mean()
    m_Df['EMASlow'] = m_Df['closing_price(dollor)'].ewm( span = m_NumSlow, min_periods = m_NumSlow - 1).mean()
    m_Df['MACD'] = m_Df['EMAFast'] - m_Df['EMASlow']
    m_Df['MACDSignal'] = m_Df['MACD'].ewm( span = m_NumSignal, min_periods = m_NumSignal-1).mean()
    m_Df['MACDDiff'] = m_Df['MACD'] - m_Df['MACDSignal']
    return m_Df


def fnBolingerBand(m_DF, n=20, k=2):
    # 볼린져 밴드
    """
    중심선: n기간 동안의 이동평균(SMA)
    상단선: 중심선 + Kσ(일반적으로 K는 2배를 많이 사용함)
    하단선: 중심선 - Kσ(일반적으로 K는 2배를 많이 사용함)
    """
    #m_DF['20d_ma'] = pd.rolling_mean(m_DF['closing_price(dollor)'], window=n)
    m_DF['Bol_upper'] = m_DF['MA20'] + k*m_DF['closing_price(dollor)'].rolling(window=n).std()
    m_DF['Bol_lower'] = m_DF['MA20'] - k*m_DF['closing_price(dollor)'].rolling(window=n).std()
    return m_DF


def data_reg(data):
    a = []
    #print(data)
    # 단위가 같은 라인끼리 정규화
    #print(data[0:1])
    #print(data[0:1]['opening_price(dollor)','closing_price(dollor)','high_price(dollor)'])

    price_df = data[['opening_price(dollor)', 'closing_price(dollor)', 'high_price(dollor)', 'low_price(dollor)', 'trade_price(dollor)', 
    'MA5', 'MA20', 'MA60', 'MA120', 'EMAFast', 'EMASlow', 'Bol_upper', 'Bol_lower']]
    price_df_min = price_df.min(axis=1).values
    price_df_max = price_df.max(axis=1).values
    price_reg_data, price_reg_min, price_reg_max = min_max_scaling(price_df, price_df_min, price_df_max)        
    

    macd_df = data[['MACD', 'MACDSignal', 'MACDDiff']]
    macd_df_min = macd_df.min(axis=1).values
    macd_df_max = macd_df.max(axis=1).values
    macd_reg_data, macd_reg_min, macd_reg_max = min_max_scaling(macd_df, macd_df_min, macd_df_max)


    sto_df = data[['Sto_K', 'Sto_D', 'Sto_SlowD']]
    sto_df_min = sto_df.min(axis=1).values
    sto_df_max = sto_df.max(axis=1).values
    sto_reg_data, sto_reg_min, sto_reg_max = min_max_scaling(sto_df, sto_df_min, sto_df_max)

    
    volume_df = data[['trade_volume']]
    volume_df_min = volume_df.min().values
    volume_df_max = volume_df.max().values
    volume_reg_data, volume_reg_min, volume_reg_max = min_max_scaling(volume_df, volume_df_min, volume_df_max)


    output_df = data[['output']]
    # output은 time+1 의 데이터이기 때문에, 현재 시점의 min max인 price 의 정보를 가져온다(실시간 test를 하기 위해)
    output_reg_data, output_reg_min, output_reg_max = min_max_scaling(output_df, price_df_min, price_df_max)

    tmp_1 = np.concatenate((price_reg_data, macd_reg_data), axis=1)
    tmp_2 = np.concatenate((tmp_1, sto_reg_data), axis=1)
    data_x = np.concatenate((tmp_2, volume_reg_data), axis=1)

    print(data_x.shape)

    data_y = output_reg_data
    print(data_y.shape)


    #print(data[data['opening_price(dollor)','closing_price(dollor)','high_price(dollor)']])
    #print(price_reg)

    return data_x, data_y, price_reg_min, price_reg_max

def data_split(seq_length, raw2p_data, save_path):
    
    del raw2p_data['opening_date(kor)'] # 데이터프레임 삭제
    del raw2p_data['closing_date(kor)'] # 위 줄과 같은 효과


    data_x, data_y, reg_min, reg_max = data_reg(raw2p_data)


    dataX = [] # 입력으로 사용될 Sequence Data
    dataY = [] # 출력(타켓)으로 사용
    for i in range(0, len(data_y) - seq_length):
        _x = data_x[i : i+seq_length]
        _y = data_y[i+1 : i+seq_length+1] # 다음 나타날 주가(정답)
        # 첫번째 행만 출력해 봄
        # if i is 0:
        #     print(_x, "->")
        #     print(_y)
        dataX.append(_x) # dataX 리스트에 추가
        dataY.append(_y) # dataY 리스트에 추가
    

    train_size = int(len(data_y) * 0.9)
    print('tr size : ', train_size)


    trX = np.array(dataX[:train_size])
    trY = np.array(dataY[:train_size])

    vaX = np.array(dataX[train_size:])
    vaY = np.array(dataY[train_size:])



    print('trX shape : ', trX.shape)
    print('trY shape : ', trY.shape)

    return trX, trY, vaX, vaY, reg_min, reg_max


def data_preprocessing(raw_dataframe):
    

    # 이동평균선 
    fnMA(raw_dataframe)

    # MACD
    fnMACD(raw_dataframe)

    # BolingerBand
    fnBolingerBand(raw_dataframe)

    # 스토캐스틱
    fnStoch(raw_dataframe)

    # output 설정
    output = np.array(raw_dataframe['closing_price(dollor)'][1:])
    output_df = DataFrame(output, columns=['output'])
    
    # output 추가하기
    r2p = pd.concat([raw_dataframe, output_df], axis=1)    
    r2p = r2p.dropna(axis=0)
    r2p.to_csv('preprocessing_data.csv',index=False) #판다스이용 csv파일로 저장

    
    #trX, trY, vaX, vaY = data_split(seq_time_step, r2p, save_path) 

    return r2p


def batch_iterator(dataX, dataY, batch_size, num_steps):

    data_len = len(dataY)
    batch_len = data_len / batch_size

    #print(data_len)
    #print(batch_len)

    epoch_size = int((batch_len) / num_steps)
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        input_x = dataX[i*batch_size: (i+1)*batch_size]
        input_y = dataY[i*batch_size: (i+1)*batch_size]
        #print(input_x.shape)
        #print(input_y.shape)
        yield (input_x, input_y)


def main():       
    raw_data = data_reader()
    seq_time_step = 14
    save_path = 'model\\'
    #tr_te_mode = True
    #tr_te_mode = False
    r2p_data = data_preprocessing(raw_data)
    #trX, trY, vaX, vaY = data_preprocessing(raw_data, seq_time_step, save_path)
    #trX, trY, vaX, vaY = data_split(seq_time_step, r2p_data, save_path) 
    data_split(seq_time_step, r2p_data, save_path) 

    #print(trX[0])

    #print(trX.shape)
    #print(trY.shape)
    


if __name__ == "__main__":
    main()
    #primary_processing()


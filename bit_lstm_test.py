import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas import DataFrame
import time

import bit_data_preprocessing as reader
 

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
tf.set_random_seed(777)

# flag 설정
flags = tf.flags
flags.DEFINE_string("save_path", "ckpt", "checkpoint_dir")
FLAGS = flags.FLAGS

#################################################################################################################################################

def info_write(input_config, output_config, infoToFile):
    
    infoToFile.write('----- Hyper parameter -----\n')
    for input_pram in input_config.__dict__:
        #print(input_pram)
        #print(input_pram.__dict__[input_pram])
        infoToFile.write(str(input_pram) + ' : ' + str(input_config.__dict__[input_pram]) + '\n')
    
    infoToFile.write('\n----- Model result -----\n')
    
    sorted_list = sorted(output_config.__dict__)
    
    for output in sorted(output_config.__dict__):
        #print(output)
        #print(output_config.__dict__[output])
        infoToFile.write(str(output) + ' : ' + str(output_config.__dict__[output]) + '\n')
    
    infoToFile.write('*'*60 + '\n')


class ModelConfig(object):
    """hyper parameter"""
    def __init__(self):
        self.input_data_column_cnt = 20  # 입력데이터의 컬럼 개수(Variable 개수)
        self.output_data_column_cnt = 1  # 결과데이터의 컬럼 개수
        
        self.seq_length = 24             # 1개 시퀀스의 길이(시계열데이터 입력 개수)
        self.rnn_cell_hidden_dim = 100   # 각 셀의 (hidden)출력 크기
        
        #self.pattern_size = 2            # 판매량 0 : 0, 판매량 1 이상 : 1
        self.forget_bias = 1.0           # 망각편향(기본값 1.0)
        
        self.num_stacked_layers = 2      # stacked LSTM layers 개수
        self.keep_prob = 1.0             # dropout할 때 keep할 비율    ## train = 0.7 test = 1.0
        
        self.batch_size = 1
        self.epoch_num = 2000             # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
        #epoch_num = 100            
        self.learning_rate = 0.01       # 학습률
        
        self.max_grad_norm = 1
        self.init_scale = 0.1


class OutputConfig(object):
    """result"""
    def __init__(self):
        self.best_iteration = 0
        self.tr_loss = 0.0
        self.processing_time = ''


class BitLstmModel(object):
    def __init__(self, is_training, config):
        self.is_training = is_training
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        
        # 텐서플로우 플레이스홀더 생성
        # 입력 X, 출력 Y를 생성한다
        
        self.X = tf.placeholder(tf.float32, [None, self.seq_length, config.input_data_column_cnt])
        #print("X: ", self.X)
        self.targets = tf.placeholder(tf.float32, [None, self.seq_length, 1])
        #print("targets: ", self.targets)
        
        # num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
        stackedRNNs = [self.lstm_cell(config) for _ in range(config.num_stacked_layers)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if config.num_stacked_layers > 1 else self.lstm_cell(config)
        
        # RNN Cell(여기서는 LSTM셀임)들을 연결
        hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
        #print("hypothesis : ", hypothesis)
        
        self.hypothesis = tf.reshape(hypothesis, [self.batch_size*self.seq_length, config.rnn_cell_hidden_dim])
        
        softmax_w = tf.get_variable("softmax_w", [config.rnn_cell_hidden_dim, config.output_data_column_cnt], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.output_data_column_cnt], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(self.hypothesis, softmax_w, softmax_b)
        
        # Reshape logits to be a 3-D tensor for sequence loss
        #self.logits = tf.reshape(logits, [self.batch_size, self.seq_length, config.output_data_column_cnt])
        self.logits = tf.reshape(logits, [self.batch_size, self.seq_length])
        #print('logits : ', self.logits)
        
        
        tgg = tf.reshape(self.targets, [self.batch_size, self.seq_length])

        # 손실함수로 평균제곱오차를 사용한다
        loss = tf.reduce_sum(tf.square(self.logits - tgg))
        """
        # regression model에서는 동작안됨(target 을 label로 받아서 int 형으로 넣어주어야 함)
        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            tgg, ##
            tf.ones([self.batch_size, self.seq_length], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
        """
        
        self.cost = loss
        self.final_state = _states
        self.one_logit = self.logits[:,-1]
        self.one_tgg = tgg[:,-1]

                
        if not is_training:
            return
        
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())
               
        
        # 모델(LSTM 네트워크) 생성
    def lstm_cell(self, config):
        # LSTM셀을 생성
        # num_units: 각 Cell 출력 크기
        # forget_bias:  to the biases of the forget gate 
        #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
        # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
        # state_is_tuple: False ==> they are concatenated along the column axis.
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=config.rnn_cell_hidden_dim, 
        forget_bias=config.forget_bias, state_is_tuple=True, activation=tf.nn.softsign)

        if config.keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
        return cell
    

def run(session, model, data_x, data_y, model_config, data_min, data_max, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    
    costs = 0.0
    iters = 0
    diff = 0.0

    ind_count = 0

    for step, (x, y) in enumerate(reader.batch_iterator(data_x, data_y, model_config.batch_size, model_config.seq_length)):

        _cost, one_logit, one_tgg = session.run([model.cost, model.one_logit, model.one_tgg], feed_dict={model.X: x, model.targets: y})

        costs += _cost
        iters += 1
        diff += np.abs(np.sum(one_logit - one_tgg))

        reg_logit = one_logit * (data_max[ind_count] - data_min[ind_count] + 1e-7) + data_min[ind_count]
        reg_tgg = one_tgg * (data_max[ind_count] - data_min[ind_count] + 1e-7) + data_min[ind_count]
        #print('logit : ', reg_logit)
        #print('tgg : ', reg_tgg)

        ind_count += 1
         
    return  diff/iters


def main():

    today_ = datetime.date.today()
    f_path = FLAGS.save_path
    
    print(f_path)

    print('test')
    model_config = ModelConfig()

    ################################################################################################

    # data road
    # upbit에서 데이터 받아와서 실시간 예측으로 수정 예정
    raw_data = reader.data_reader()
    dataX, dataY, data_min, data_max = reader.data_preprocessing(raw_data, model_config.seq_length)

    ################################################################################################    
    
    sess = tf.Session()
        
    m_te = BitLstmModel(is_training=False, config=model_config)

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(f_path)


    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            

    if ckpt and ckpt.model_checkpoint_path:
        print('ckpt : ', ckpt)
        print('ckpt.model_checkpoint_path : ' , ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print ("No checkpoint file found")

        
    te_diff = run(sess, m_te, dataX, dataY, model_config, data_min, data_max)
    #print(te_diff)


if __name__ == "__main__":
    main()


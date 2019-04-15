# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:04:19 2019

@author: sumedh
"""
import os
import load_data as ld
import w2v_features as w2v
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras import backend as K
from keras.layers import Dense, LSTM, Input, Add, TimeDistributed, Flatten, RepeatVector, merge, Lambda, Multiply, Reshape, Permute
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import RMSprop


def encoder_decoder(data, en_shape, de_shape, hidden_units, learning_rate, clip_norm, epochs, batch_size):
    ''' encoder '''
    encoder_inputs = Input(shape=en_shape)
    
    encoder_LSTM = LSTM(hidden_units, dropout_U = 0.02, dropout_W = 0.02 ,return_state=True)
    encoder_LSTM_rev=LSTM(hidden_units,return_state=True,go_backwards=True)
    
    encoder_outputsR, state_hR, state_cR = encoder_LSTM_rev(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)
    
    state_hfinal=Add()([state_h,state_hR])
    state_cfinal=Add()([state_c,state_cR])
    
    encoder_states = [state_hfinal,state_cfinal]
    
    ''' decoder '''
    decoder_inputs = Input(shape=(None,de_shape[1]))
    decoder_LSTM = LSTM(hidden_units,return_sequences=True,return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states) 
    decoder_dense = Dense(de_shape[1],activation='linear')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    
    model.compile(loss='mse',optimizer=rmsprop)

    x_train,x_test,y_train,y_test=tts(data["review"],data["summaries"],test_size=0.20)
    model.fit(x=[x_train,y_train],
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test,y_test], y_test))
    
    ''' inference mode '''
    encoder_model_inf = Model(encoder_inputs,encoder_states)
    
    decoder_state_input_H = Input(shape=(hidden_units,))
    decoder_state_input_C = Input(shape=(hidden_units,)) 
    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs,
                                                                     initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)
    
    #scores = model.evaluate([x_test,y_test],y_test, verbose=0)
    
    return model,encoder_model_inf,decoder_model_inf


def summarize(review, en_shape, de_shape, encoder, decoder,max_len = 300):
    stop_pred = False
    review =  np.reshape(review, (1, en_shape[0], en_shape[1]))
    
    init_state_val = encoder.predict(review)
    target_seq = np.zeros((1,1,300))
    
    generated_summary=[]
    while not stop_pred:
        decoder_out,decoder_h,decoder_c= decoder.predict(x=[target_seq]+init_state_val)
        generated_summary.append(decoder_out)
        init_state_val= [decoder_h,decoder_c]
        target_seq=np.reshape(decoder_out,(1,1,300))
        if len(generated_summary)== max_len:
            stop_pred=True
            break
    
    #sent_vecs = np.reshape(generated_summary, de_shape)
    summ = ''
    for k in generated_summary:
       # summ = summ+((getWord(k)[0]+' ') if getWord(k)[1]>0.2 else '')
       summ = summ + ohe.inverse_transform([np.argmax(k)])[0].strip()+" "
    
    return summ





path = os.getcwd()
reviews = path+'\\summaries handwritten\\'
datasets={'reviews':reviews}
data_categories=["training","validation","test"]
filenames= ld.load_data(datasets['reviews'],data_categories[0])
data = ld.make_data_dict(filenames,datasets,data_categories)
corp = w2v.create_corpus(data)
embed_model = w2v.word2vec_model(google = True)
ohe, onehot_encoded,onehot = w2v.one_hot_encode(corpus = corp, vocab = embed_model.index2word)
train_data = w2v.w2v_matrix(embed_model,data)
train_data= w2v.cut_seq(train_data,10,5)
train_data["summaries"]=np.array(train_data["summaries"])
train_data["review"]=np.array(train_data["review"])
train_data["summaries"]=np.array(list(map(w2v.addones,train_data["summaries"])))


trained_model,encoder,decoder = encoder_decoder(data = train_data,
                en_shape = train_data['review'][0].shape,
                de_shape = train_data['summaries'][0].shape,
                hidden_units = 500,
                learning_rate = 0.005,
                clip_norm = 0,
                epochs = 10,
                batch_size = 10)


print(summarize(review = train_data["review"][4],
                             en_shape = train_data['review'][4].shape,
                             de_shape = train_data['summaries'][4].shape,
                             max_len = 5,
                             encoder = encoder, 
                             decoder = decoder))

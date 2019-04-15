# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:18:26 2019

@author: sumedh
"""


def encoder_decoder_attention(data, ip_shape, op_shape, units, lr, ep, bz, c_norm):
    
    '''encoder'''
    encoder_inputs = Input(shape=(ip_shape))
    encoder_LSTM = LSTM(units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,return_state=True)
    encoder_LSTM_rev=LSTM(units,return_state=True,return_sequences=True,dropout_U=0.05,dropout_W=0.05,go_backwards=True)
    
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)
    encoder_outputsR, state_hR, state_cR = encoder_LSTM_rev(encoder_inputs)
    
    state_hfinal=Add()([state_h,state_hR])
    state_cfinal=Add()([state_c,state_cR])
    encoder_outputs_final = Add()([encoder_outputs,encoder_outputsR])
    
    encoder_states = [state_hfinal,state_cfinal]
    
    '''decoder'''
    decoder_inputs = Input(shape=(None,op_shape[1]))
    decoder_LSTM = LSTM(units,return_sequences=True,dropout_U=0.2,dropout_W=0.2,return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states)
    
    
    '''attention'''
    attention = TimeDistributed(Dense(1, activation = 'tanh'))(encoder_outputs_final)
    attention = Flatten()(attention)
    attention = Multiply()([decoder_outputs, attention])
    attention = Activation('softmax')(attention)
    attention = Permute([2, 1])(attention)
 
    decoder_dense = Dense(op_shape[1],activation='softmax')
    decoder_outputs = decoder_dense(attention)
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    
    rmsprop = RMSprop(lr=lr)#,clipnorm=c_norm)
    model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
    
    x_train,x_test,y_train,y_test=tts(data["review"],data["summaries"],test_size=0.20)
    history= model.fit(x=[x_train,y_train],
              y=y_train,
              batch_size=bz,
              epochs=ep,
              verbose=1,
              validation_data=([x_test,y_test], y_test))
    
    
    """_________________inference mode__________________"""
    encoder_model_inf = Model(encoder_inputs,encoder_states)
    
    decoder_state_input_H = Input(shape=(ip_shape[0],))
    decoder_state_input_C = Input(shape=(ip_shape[0],)) 
    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs,
                                                                     initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)
    
    scores = model.evaluate([x_test,y_test],y_test, verbose=1)
    
    
    print('LSTM test scores:', scores)
    print(model.summary())
    return model,encoder_model_inf,decoder_model_inf,history



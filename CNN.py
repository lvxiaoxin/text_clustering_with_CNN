from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam


def CNN(MAX_NB_WORDS, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, TARGET_DIM=300, EMBEDDING_TRAINABLE=False):
    embedding_matrix_copy = embedding_matrix.copy()
    
    # 1st layer - Embedding layer
    pretrained_embedding_layer = Embedding(
        input_dim=MAX_NB_WORDS,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
    )

    # apply 1st layer
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = pretrained_embedding_layer(sequence_input)

    # 2nd layer - Conv layer
    x = Conv1D(300, 5, activation='tanh', padding='same')(embedded_sequences)
    
    # 3nd layer - Max pooling 
    x = GlobalMaxPooling1D()(x)

    # Dropout, Loss cal, fine-tune
    x = Dropout(0.5)(x)
    predictions = Dense(TARGET_DIM, activation='sigmoid')(x)
    model = Model(sequence_input, predictions)
    model.layers[1].trainable = EMBEDDING_TRAINABLE
    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['mae'])
    
    model.summary()
    return model

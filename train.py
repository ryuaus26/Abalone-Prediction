import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from convokit import Corpus,download

# config params
VOCAB_SIZE = 8500
OUTPUT_SEQUENCE_LENGTH = 40
MAX_SAMPLES = 50000
BUFFER_SIZE = 2048
BATCH_SIZE = 64
EPOCHS = 200
EMBED_DIM = 256
LATENT_DIM = 1024
NUM_HEADS = 8
START_TOKEN = '[start]'
STOP_TOKEN = '[stop]'






def load_conversations(max_samples:int=50000) -> list:
    def clean_text(input_text:str) -> str:
        res = input_text.lower().strip()
        res = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", res)
        return res
        
    sep = ' +++$+++ '

    movie_lines = {}
    with open('movie_lines.txt', 'r', encoding='iso-8859-1') as f:
        lines = f.read().split('\n')
    for line in lines:
        key = line.split(sep)[0]
        value = line.split(sep)[-1]
        movie_lines[key] = value
    
    line_pairs = []
    with open('movie_conversations.txt', 'r', encoding='iso-8859-1') as f:
        lines = f.read().split('\n')
    for line in lines:
        conversation = line.split(sep)[-1][1:-2].replace("'", '').split(', ')
        for i in range(len(conversation) - 1):
            statement = clean_text(movie_lines[conversation[i]])
            response = clean_text(movie_lines[conversation[i + 1]])
            response = START_TOKEN + ' ' + response + ' ' + STOP_TOKEN
            line_pairs.append((statement, response))
            if len(line_pairs) >= max_samples:
                return line_pairs
    return line_pairs

line_pairs = load_conversations(MAX_SAMPLES)

def split_train_test_data(dataset:list, test_size:float=0.1) -> tuple:
    np.random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - test_size))
    train_ds = dataset[:split_idx]
    valid_ds = dataset[split_idx:]
    return train_ds, valid_ds

train_pairs, valid_pairs = split_train_test_data(line_pairs)

def get_vectorizer(dataset:list) -> tuple:
    input_vectorizer = layers.TextVectorization(
        VOCAB_SIZE,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
        standardize=None
    )
    target_vectorizer = layers.TextVectorization(
        VOCAB_SIZE,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH + 1,
        standardize=None
    )
    statements, responses = zip(*dataset)
    input_vectorizer.adapt(list(statements))
    target_vectorizer.adapt(list(responses))
    return input_vectorizer, target_vectorizer

input_vectorizer, target_vectorizer = get_vectorizer(line_pairs)

def create_dataset(dataset:list):
    def vectorize_text(statements, responses):
        inputs, outputs = input_vectorizer(statements), target_vectorizer(responses)
        return (
            {"encoder_inputs": inputs, "decoder_inputs": outputs[:, :-1]},
            {"outputs": outputs[:, 1:]}
        )

    statements, responses = zip(*dataset)
    dataset = tf.data.Dataset.from_tensor_slices((list(statements), list(responses)))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(vectorize_text)
    return dataset.shuffle(BUFFER_SIZE).prefetch(16).cache()

train_ds = create_dataset(train_pairs)
valid_ds = create_dataset(valid_pairs)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"), 
            layers.Dense(embed_dim)
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
            
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(length)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential([
            layers.Dense(latent_dim, activation="relu"), 
            layers.Dense(embed_dim)
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs,key=encoder_outputs,
            attention_mask=padding_mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        print(i, j)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

def create_model() -> keras.Model:
    # encoder
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    encoder_augmented_inputs = PositionalEmbedding(OUTPUT_SEQUENCE_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
    encoder_outputs = TransformerEncoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(encoder_augmented_inputs)

    # decoder
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")
    decoder_augmented_inputs = PositionalEmbedding(OUTPUT_SEQUENCE_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
    decoder_outputs = TransformerDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(decoder_augmented_inputs, encoded_seq_inputs)
    decoder_outputs = layers.Dropout(0.5)(decoder_outputs)
    decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(decoder_outputs)

    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs, name='outputs')
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])

    model = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )

    model.compile(
        "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model

model = create_model()

history = model.fit(
    train_ds, 
    validation_data=valid_ds,
    epochs=EPOCHS 
)

# Save model
model.save_weights('transformer_chatbot.h5')


model.load_weights('transformer_chatbot.h5')

vocab = target_vectorizer.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))

def decode_sequence(input_sentence):
    tokenized_input_sentence = input_vectorizer([input_sentence])
    decoded_sentence = START_TOKEN
    for i in range(OUTPUT_SEQUENCE_LENGTH):
        tokenized_target_sentence = target_vectorizer([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        decoded_sentence += ' ' + sampled_token

        if sampled_token == STOP_TOKEN:
            break
    return decoded_sentence

def response(input_text):
    input_sentence = input_text
    translated = decode_sequence(input_sentence)
    print('-'*50)
    print('Input: ', input_sentence)
    print('Output: ', translated)
    

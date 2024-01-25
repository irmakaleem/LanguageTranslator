import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LSTM_NODES = 256
NUM_SENTENCES = 20000
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 20000
EMBEDDING_SIZE = 100

data = [("the future is generative ai", "مستقبل جنر ی ٹو اے آئ ی ہے")]

# Preprocess and tokenize the data
input_sentences = []
output_sentences = []
output_sentences_inputs = []

for line in data:
    input_sentence, output = line
    output_sentence = output + ' <eos>'
    output_sentence_input = '<sos> ' + output

    input_sentences.append(input_sentence)
    output_sentences.append(output_sentence)
    output_sentences_inputs.append(output_sentence_input)

input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

word2idx_inputs = input_tokenizer.word_index
print('Total unique words in the input: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
print("Length of longest sentence in input: %g" % max_input_len)

output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

word2idx_outputs = output_tokenizer.word_index
print('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: %g" % max_out_len)

encoder_inputs = pad_sequences(input_integer_seq, maxlen=max_input_len)
print("encoder_inputs.shape:", encoder_inputs.shape)
print("encoder_inputs[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
print("decoder_inputs[0]:", decoder_inputs[0])
print("decoder_inputs.shape:", decoder_inputs.shape)

decoder_targets = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')

# One-hot encode the target sequences
decoder_targets_one_hot = to_categorical(decoder_targets, num_words_output)

# Create the Encoder network
encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = Embedding(MAX_NUM_WORDS, LSTM_NODES)(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]

# Create the Decoder network
decoder_inputs_placeholder = Input(shape=(max_out_len,))

decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile the model
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
predicted_sequences = []  # Accumulate predicted sequences for each epoch

for epoch in range(EPOCHS):
    model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot, batch_size=BATCH_SIZE, epochs=1, verbose=1)

    # Print the predicted output for a sample English sentence
    sample_input = encoder_inputs[0:1]
    predicted_output = model.predict([sample_input, decoder_inputs[0:1]])
    predicted_sequence = [tf.argmax(output, axis=-1).numpy() for output in predicted_output][0]
    predicted_sequences.append(predicted_sequence)

# Function to convert sequence to text
def sequence_to_text(sequence, tokenizer):
    reverse_word_index = dict(map(reversed, tokenizer.word_index.items()))
    return ' '.join([reverse_word_index.get(i, '') for i in sequence])

# Print input, target, and predicted values after all epochs
for epoch in range(EPOCHS):
    input_text = sequence_to_text(encoder_inputs[0], input_tokenizer)
    target_text = sequence_to_text(decoder_targets[0], output_tokenizer)
    predicted_text = sequence_to_text(predicted_sequences[epoch], output_tokenizer)

    print(f'Epoch {epoch + 1}/{EPOCHS}, Input: {input_text}, Target: {target_text}, Predicted: {predicted_text}')
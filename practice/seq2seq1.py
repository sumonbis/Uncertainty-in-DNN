import tensorflow as tf
import numpy as np
import unicodedata
import re
from practice.Encoder import Encoder
from practice.Decoder import Decoder

raw_data = (
    ('What a ridiculous concept!', 'Ki ajaira kotha !'),
    ('Your idea is not entirely crazy.', "Tumar chinta ta ekdom kharap na."),
    ("A man's worth lies in what he is.", "Ekjon manusher mullo tar nijer moddhei nihito."),
    ('What he did is very wrong.', "Se ja korche kharap korche."),
    ("All three of you need to do that.", "Tumra tin joner eta kora uchit."),
    ("Are you giving me another chance?", "Amake ar ekta sujog diccho ?"),
    ("Both Tom and Mary work as models.", "Tom ar mary duijon e nayika hisebe kaj kore."),
    ("Can I have a few minutes, please?", "Amake kichu somoy diba, doya kore ?"),
    ("Could you close the door, please?", "Dorja ta bondho korba, doya kore ?"),
    ("Did you plant pumpkins this year?", "Misti kumra gach lagaicho naki ei bochor ?"),
    ("Do you ever study in the library?", "Tumi kokhno library te porasuna korcho ?"),
    ("Don't be deceived by appearances.", "Cehara dekhe dhoka kheyo na."),
    ("Excuse me. Can you speak English?", "Maf koro. Tumi english bolte parba?"),
    ("Few people know the true meaning.", "Khub kom lok asol ortho jane."),
    ("Germany produced many scientists.", "Germany onek biggani tairi korche."),
    ("Guess whose birthday it is today.", "Onuman koro kar jonmodin ajke !"),
    ("He acted like he owned the place.", "She emon ekta vab niche jeno jaiga ta tar."),
    ("Honesty will pay in the long run.", "Sotota valo puroskar niye ashe ek somoy."),
    ("How do we know this isn't a trap?", "Amra kivabe janbo eta ekta fad na ?"),
    ("I can't believe you're giving up.", "Ami bisshas korte partechi na tumi hal chere diccho."),
)

# raw_data = (
#     ('compile fit predict', 'compile fit predict'),
#     ('compile fit save', "compile fit save"),
#     ("add loadweights", "add loadweights"),
#     ('pop add Model', "pop add Model"),
#     ("dense BatchNormalization Activation", "dense BatchNormalization Activation"),
#     ("compile saveweights loadweights", "compile saveweights loadweights"),
#     ("loadweights add Model", "loadweights add Model"),
#     ("backend clearsession", "backend clearsession"),
#     ("VGG16 Dropout Model", "VGG16 Dropout Model"),
#     ("fit clearsession", "fit clearsession"),
#     ("Sequential Concatenate", "Sequential Concatenate"),
#     ("fit predictclasses", "fit predictclasses"),
#     ("loadmodel add", "loadmodel add"),
#     ("saveweights loadmodel", "saveweights loadmodel"),
#     ("loadmodel setvalue", "loadmodel setvalue"),
#     ("dense sigmoid compile", "dense sigmoid compile"),
#     ("dense softmax compile fit", "dense softmax compile fit"),
#     ("dense timedistributed", "dense timedistributed"),
#     ("read preprocess add compile fit predict", "read preprocess add compile fit predict"),
#     ("read preprocess add compile fit savemodel", "read preprocess add compile fit savemodel"),
# )

def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_states = en_states

        de_outputs = decoder(target_seq_in, de_states)
        logits = de_outputs[0]
        loss = loss_func(target_seq_out, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss
def predict():
    test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    test_source_text="you plant wrong"
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []

    while True:
        de_output, de_state_h, de_state_c = decoder(
            de_input, (de_state_h, de_state_c))
        de_input = tf.argmax(de_output, -1)
        a=de_input.numpy()[0][0]
        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))

raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)

raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                        padding='post')

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

# ATTENTION: always finish with fit_on_texts before moving on
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                           padding='post')

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                            padding='post')

dataset = tf.data.Dataset.from_tensor_slices(
    (data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(20).batch(5)


EMBEDDING_SIZE = 32
LSTM_SIZE = 64

en_vocab_size = len(en_tokenizer.word_index) + 1
encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

fr_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

optimizer = tf.keras.optimizers.Adam()

NUM_EPOCHS = 650
BATCH_SIZE = 5

for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)

    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out, en_initial_states)

    print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))

    # try:
    #      predict()
    # except Exception:
    #     continue

predict()
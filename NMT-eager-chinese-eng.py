# -*- coding: utf-8 -*-
# https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
import unicodedata 
import re 
import numpy as np 
import os 
import time 
import tensorflow as tf 

tf.enable_eager_execution() 

path_to_file = "data/eng-chinese.txt" 

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)\
        if unicodedata.category(c) != 'Mn')


def preprocess_eng(w):
    w = unicode_to_ascii(w.lower().strip())    
    w = re.sub(r"([?.!,¿])", r" \1 ", w)    # 标点符号和文字之间添加空格
    w = re.sub(r'[" "]+', " ", w)           # 将多个空格转换为1个空格
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)  # 去除非合法的字符
    w = w.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def preprocess_chinese(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r'[" "]+', "", w)   
    w = w.rstrip().strip()
    w = " ".join(list(w))
    w = '<start> ' + w + ' <end>'
    return w

# 读入数据, 返回一个list，其中元素是 (语言1，语言2) 的元组
def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[w for w in l.split('\t')]  for l in lines[:num_examples]] 
    word_pairs = [[preprocess_eng(w[0]), preprocess_chinese(w[1])] for w in word_pairs]
    return word_pairs

# 构建词和编号之间的对应关系. 构建2个字典
# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
    
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        
        self.vocab = sorted(self.vocab)
        
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1
        
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def max_length(tensor):
    # 后续需要将所有的句子长度填充值 max_length
    return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # 将中文作为源语言，因为作为目标语言
    inp_lang = LanguageIndex(ch for en, ch in pairs)      # 初始化两个不同的类
    targ_lang = LanguageIndex(en for en, ch in pairs)

    # for k, v in targ_lang.word2idx.items():
    #     print(k, v)

    # Vectorize the input and target languages
    
    # 中文句子. 一个 num_examples 长的list,每个元素是一个list，代表一个句子.元素是整形
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]
    
    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    # 46, 38

    # 在 input_tensor 中每个元素的后方填充0，扩充到 max_length_inp 长
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')    
    # 在 target_tensor 中每个元素的后方填充0，扩充到 max_length_tar 长
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar



# Try experimenting with the size of that dataset
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.05)

# Show length  24000 24000 6000 6000
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

# tf.data
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)    # 9394
vocab_tar_size = len(targ_lang.word2idx)   # 4918

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


def gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    # if tf.test.is_gpu_available():
    #   return tf.keras.layers.CuDNNGRU(units, 
    #                                   return_sequences=True, 
    #                                   return_state=True, 
    #                                   recurrent_initializer='glorot_uniform')
    # else:
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        # vacab_size=vocab_inp_size=9394, embedding_dim=256 enc_units=1024 batch_sz=64
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        # x 是训练数据，shape=(64，16). 代表一个批量共计64个句子，每个句子长度为16
        # hidden 是隐含层状态，shape=(64,1024)
        x = self.embedding(x)                                  # x.shape = (64,16,256)
        output, state = self.gru(x, initial_state = hidden)    # output.shape=(64,16,1024), state.shape=(64,1024)
        return output, state                                   # output是整个seq的输出，state是最后一个时间步的state
    
    def initialize_hidden_state(self):
        # 返回 hidden代表隐含层状态.  shape=(64,1024)
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        # vocab_size=4918, embedding_dim=256, dec_units=1024, batch_sz=64
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)   # 输出节点个数为单词个数，后接softmax
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # 该函数执行 decoder 中一步attention，输出一步的结果
        # 第一个时间步的hidden初始化为encoder的最后输出state，随后hidden在decoder中传递

        # enc_output shape == (batch_size, max_length, hidden_size)
        # hidden.shape == (batch_size, hidden size)  (64,1024)
        # hidden_with_time_axis.shape == (batch_size, 1, hidden size) 扩展一个维度
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, hidden_size)
        # enc_output代表encoder执行后的输出结果. hidden_with_time_axis需要和enc_output中每个元素计算score，随后规约作为权重
        # self.W1(enc_output).shape=(64, 16, 1024), self.W2(hidden_with_time_axis).shape=(64, 1, 1024)       
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # element_wise 加权.  (batch_size, max_length, 1) * (batch_size, max_length, hidden_size)
        context_vector = attention_weights * enc_output   
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x是当前时间步的一个批量的输入词, 需要经过 embedding 处理
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # 将输入词 x 和 context_vector 的最后一维连接起来，作为 GRU 的输入
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU 
        # GRU在当前1个时间步的输出
        output, state = self.gru(x)   # output = (64,1,1024), state=(64,1024)
        output = tf.reshape(output, (-1, output.shape[2]))   # output.shape=(batch_size * 1, hidden_size=1024)
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)       # 输出为 vocab 维度，用于后续 softmax 计算损失
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)


optimizer = tf.train.AdamOptimizer()


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)   # real=0时表示该位置时填充上去的，没有字符
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


checkpoint_dir = 'data/chinese-eng'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# 如果训练，此处设置 10
EPOCHS = 0

for epoch in range(EPOCHS):
    start = time.time()
    
    hidden = encoder.initialize_hidden_state()
    total_loss = 0
    
    for (batch, (inp, targ)) in enumerate(dataset):
        # inp.shape=(64,16).  targ.shape=(64,11)
        # print(targ.shape)
        # print(targ[0])
        loss = 0
        with tf.GradientTape() as tape:
            # enc_output.shape=(64,16,1024), enc_hidden.shape=(64,1024)
            enc_output, enc_hidden = encoder(inp, hidden)
            # 将 decoder 的隐含变量初始化为 enc_hidden
            dec_hidden = enc_hidden
            
            # 初始化输入，第一个时间步的输入为 <start>
            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)   # shape=(64,1)
            
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):          # decoder 训练11个时间步，输出11长度的句子
                # passing enc_output to the decoder
                # predictions.shape=(64,词个数). 用于后接 softmax 输出.  dec_hidden.shape=(64,1024)
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)  # dec_hidden是保留的

                # 调用loss函数，计算在第 t 个时间步的损失
                loss += loss_function(targ[:, t], predictions)
                
                # using teacher forcing  用真实值作为下一个时间步的输入
                dec_input = tf.expand_dims(targ[:, t], 1)
        
        print("loss:", loss)
        batch_loss = (loss / int(targ.shape[1]))  # 除以 时间步
        
        total_loss += batch_loss
        
        variables = encoder.variables + decoder.variables
        
        gradients = tape.gradient(loss, variables)
        # gradients = tf.clip_by_global_norm(gradients, 5).  # 该步骤会报错
        optimizer.apply_gradients(zip(gradients, variables))

        # 测试
        # translate('你真的很厉害。', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

        # print("-----\n\n")
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):

    # sentence是输入的句子，encoder和decoder是model, inp_lang, targ_lang分别是输入和输出的类，存储词和编号的对应字典.
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    sentence = preprocess_chinese(sentence)
    # print(sentence)

    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')  #(1,16)
    inputs = tf.convert_to_tensor(inputs)
    # print(inputs)
    
    result = ''

    hidden = [tf.zeros((1, units))]                   # (1,64)
    enc_out, enc_hidden = encoder(inputs, hidden)     # (1, 16, 1024) (1, 1024)

    dec_hidden = enc_hidden                           # (1, 1024)
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))  # (16,)
        attention_plot[t] = attention_weights.numpy()
        # print(attention_weights)

        # 根据伯努利分布，根据输出的概率，进行采样. 作为下一个时间步的输入
        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    old_sentence = sentence
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        
    print('Input: {}'.format(old_sentence))
    print('Predicted translation: {}'.format(result))
    
    # 去除由于延长句子而产生的为空的区域
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]

    # 绘图保存
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))
    plt.savefig("data/chinese-eng-jpg/"+old_sentence+".jpg")


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore("data/chinese-eng/ckpt-50")

#
translate('我相信他明天會來。', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
print("---\n")
translate('如果你不去音樂會了, 我也不去。', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
print("---\n")
translate('我不是傻瓜。', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
print("---\n")
translate('有没有一个国家比美国更提倡爱国主义？', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
print("---\n")
translate('即使是现在，我偶尔还是想见到你。不是今天的你，而是我记忆中曾经的你。', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
print("---\n")
translate('我已經完成我的作業。', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
print("---\n")
translate('令我吃惊的是，他很容易就想出了一个方案。', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)




# coding: utf-8

# # TV Script Generation
# # 电视剧脚本生成
# 
# In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
# 
# 在这个项目中，您将使用RNN生成自己的[Simpsons](https://en.wikipedia.org/wiki/The_Simpsons)电视剧本。 您将使用27季剧本[Simpsons数据集](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data)的一部分。 你将建立的神经网络将在[Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern)上为一个场景生成一个新的电视剧本。
# 
# ## Get the Data
# ## 获取数据
# 
# The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..
# 
# 数据已经为您提供。 您将使用原始数据集的一个子集。 它只包含在Moe的小酒馆的场景。 这不包括其他版本的小酒馆，如 "Moe's Cavern", "Flaming Moe's"， "Uncle Moe's Family Feed-Bag"等。

# In[1]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
print(len(text))


# ## Explore the Data
# ## 探索数据
# 
# Play around with `view_sentence_range` to view different parts of the data.
# 
# 使用`view_sentence_range`浏览数据的不同部分。

# In[2]:


view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# ## Implement Preprocessing Functions
# ## 实现预处理函数
# 
# The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
# 
# 对任何数据集首先要做的事就是预处理。 实现下面的预处理功能：
# 
# - Lookup Table
# - 查找表
# - Tokenize Punctuation
# - 标点符号token化
# 
# ### Lookup Table
# ### 查找表
# 
# To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
# 
# 要创建一个单词嵌入，您首先需要将单词转换为id。 在这个函数中，创建两个字典：
# 
# - Dictionary to go from the words to an id, we'll call `vocab_to_int`
# - 使word转换到id的字典，我们称之为`vocab_to_int`
# - Dictionary to go from the id to word, we'll call `int_to_vocab`
# - 使id转换为word的字典，我们称之为`int_to_vocab`
# 
# Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`
# 
# 以元组`（vocab_to_int，int_to_vocab）`的形式的返回这些字典

# In[3]:


import numpy as np
import problem_unittests as tests

from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    words_count = Counter(text)
    vocab = words_count.most_common()
    vocab_to_int = {word[0]:idx  for idx,word in enumerate(vocab) }
    int_to_vocab = {idx: word  for word, idx in vocab_to_int.items()}
    
    return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


# ### Tokenize Punctuation
# ### 标点符号token化
# 
# 
# We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".
# 
# 我们将使用空格作为分隔符将脚本分割成一个单词数组。 然而，句号和惊叹号等标点使得神经网络很难区分“再见”和“再见！”这两个词。
# 
# Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
# 
# 实现`token_lookup`函数并返回一个字典，用来将符号像"!"  转换为"||Exclamation_Mark||"。 为符号为关键字且值为令牌的以下符号创建字典：
# 
# - Period ( . )
# - Comma ( , )
# - Quotation Mark ( " )
# - Semicolon ( ; )
# - Exclamation mark ( ! )
# - Question mark ( ? )
# - Left Parentheses ( ( )
# - Right Parentheses ( ) )
# - Dash ( -- )
# - Return ( \n )
# 
# This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".
# 
# 这个词典将被用来标记符号并在其周围添加分隔符（空格）。 这将符号分离为自己的单词，使得神经网络更容易预测下一个单词。 确保你不要使用可能被混淆的单词。 不要使用标记“dash”，可以尝试使用"||dash||"之类的东西。

# In[4]:


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punc_to_token = {
        '.' : '||Period||',
        ',' : '||Comma||',
        '"' : '||Quotation_Mark||',
        ';' : '||Semicolon||',
        '!' : '||Exclamation_mark||',
        '?' : '||Question_mark||',
        '(' : '||Left_Parentheses||',
        ')' : '||Right_Parentheses||',
        '--' : '||Dash||',
        '\n' : '||Return||'
    }
    
    return punc_to_token

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)


# ## Preprocess all the data and save it
# ## 预处理所有数据并保存
# 
# Running the code cell below will preprocess all the data and save it to file.
# 
# 运行下面的代码单元将预处理所有数据并将其保存到文件。

# In[5]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


# # Check Point
# # 检查点
# 
# This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.
# 
# 这是你的第一个检查点。 如果您决定回来这台笔记本，或不得不重新启动笔记本，你可以从这里开始。 预处理的数据已保存到磁盘。

# In[6]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


# ## Build the Neural Network
# ## 构建神经网络
# 
# You'll build the components necessary to build a RNN by implementing the following functions below:
# 
# 您将通过执行以下功能来构建构建RNN所需的组件：
# 
# - get_inputs
# - get_init_cell
# - get_embed
# - build_rnn
# - build_nn
# - get_batches
# 
# ### Check the Version of TensorFlow and Access to GPU
# ### 检查TensorFlow的版本并GPU访问权限

# In[7]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# ### 输入
# 
# Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
# 
# 实现`get_inputs（）`函数并为神经网络创建TF占位符。 它应该创建以下占位符：
# 
# - Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
# - Targets placeholder
# - Learning Rate placeholder
# 
# Return the placeholders in the following tuple `(Input, Targets, LearningRate)`
# 
# 以元组的形式`（Input，Targets，LearningRate）`返回占位符

# In[8]:


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    Input = tf.placeholder(tf.int32, [None, None], name="input")
    Targets = tf.placeholder(tf.int32, [None, None], name="target")
    LearningRate = tf.placeholder(tf.float32, name="learning_rate")
    
    return (Input,Targets,LearningRate)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)


# In[9]:


def get_keep_prob():
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return keep_prob


# ### Build RNN Cell and Initialize
# ### 构建RNN单元并初始化
# 
# Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
# 
# 将一个或多个[`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell)添加到[`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell)。
# 
# - The Rnn size should be set using `rnn_size`（Rnn大小应该使用`rnn_size`设置）
# - Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function（使用MultiRNNCell函数 [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state)初始化单元状态
# - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)（使用[`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)将名称“initial_state”应用于初始状态）
# 
# Return the cell and initial state in the following tuple `(Cell, InitialState)`
# 
# 在下面的元组`（Cell，InitialState）`中返回单元格和初始状态

# In[10]:


def get_init_cell(batch_size, rnn_size, keep_probablity=None):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm_layers = 1
    
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    if keep_probablity == None :
        keep_probablity = get_keep_prob()
    
    dorp = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_probablity)
    
    cells = tf.contrib.rnn.MultiRNNCell([dorp]*lstm_layers)
    
    init_state = tf.identity(cells.zero_state(batch_size, tf.float32), name="initial_state")  # lstm的zero_state与batch_size有什么关系？
    return (cells, init_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)


# ### Word Embedding
# ### 单词嵌入
# 
# Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.
# 
# 使用TensorFlow将嵌入应用于`input_data`。 返回嵌入的序列。

# In[11]:


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), minval=-1, maxval=1)) 
    embed = tf.nn.embedding_lookup(embedding, input_data)
    
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)


# ### Build RNN
# ### 构建RNN
# 
# You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
# 
# 您在`get_init_cell（）`函数中创建了一个RNN单元。 是时候使用单元来创建一个RNN了。
# 
# - Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)（使用[`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)构建RNN）
# - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)（使用 [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)将名称"final_state"用于最终状态）
# 
# Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 
# 
# 在下面的元组（`Outputs，FinalState）`中返回输出和final_state状态

# In[12]:


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    
    Outputs, Final_State = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    Final_State = tf.identity(Final_State, name="final_state")
    return Outputs, Final_State


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)


# ### Build the Neural Network
# ### 构建神经网络
# 
# Apply the functions you implemented above to:
# 
# 将以上实现的功能应用于：
# 
# - Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.（使用`get_embed（input_data，vocab_size，embed_dim）`函数将嵌入应用于`input_data`。）
# - Build RNN using `cell` and your `build_rnn(cell, inputs)` function.（使用`cell`和你的`build_rnn（cell，inputs）`函数建立RNN。）
# - Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.（应用线性激活函数的全连接图层，使用`vocab_size`作为输出的数量。）
# 
# Return the logits and final state in the following tuple (Logits, FinalState) 
# 
# 返回下列元组中的logits和final状态（Logits，FinalState）

# In[13]:


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    embed = get_embed(input_data, vocab_size, embed_dim)
    
    outputs, final_state = build_rnn(cell, embed)

    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
  
    return logits, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)


# ### Batches
# ### 分批
# 
# Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
# 
# 使用`int_text`实现`get_batches`来创建批量的输入和目标。 批次应该是一个形状为`(number of batches, 2, batch size, sequence length)`的Numpy数组。 每个批次包含两个元素：
# 
# - The first element is a single batch of **input** with the shape `[batch size, sequence length]`
# - The second element is a single batch of **targets** with the shape `[batch size, sequence length]`
# 
# If you can't fill the last batch with enough data, drop the last batch.
# 
# 如果不能用足够的数据填充最后一批，请删除最后一批。
# 
# For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2, 3)` would return a Numpy array of the following:
# ```
# [
#   # First Batch
#   [
#     # Batch of Input
#     [[ 1  2  3], [ 7  8  9]],
#     # Batch of targets
#     [[ 2  3  4], [ 8  9 10]]
#   ],
#  
#   # Second Batch
#   [
#     # Batch of Input
#     [[ 4  5  6], [10 11 12]],
#     # Batch of targets
#     [[ 5  6  7], [11 12 13]]
#   ]
# ]
# ```

# In[14]:


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    
    words_per_batch = batch_size * seq_length
    
    n_batches = len(int_text) // words_per_batch
    
    arr = np.array(int_text[:words_per_batch * n_batches])
    arr = arr.reshape([batch_size,-1])
    
    batches =  []
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x)
        if n+seq_length >= arr.shape[1]:
            idx = 0
        else:
            idx = n+seq_length
        y[:,:-1],y[:, -1]  = x[:, 1:], arr[:, idx]
        batches.append([x,y])
        
    batches = np.array(batches)
    return batches



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)


# ## Neural Network Training
# ## 神经网络的训练
# ### Hyperparameters
# ### 超参数
# Tune the following parameters:
# 
# 调整下面的参数：
# 
# - Set `num_epochs` to the number of epochs.
# - Set `batch_size` to the batch size.
# - Set `rnn_size` to the size of the RNNs.
# - Set `embed_dim` to the size of the embedding.
# - Set `seq_length` to the length of sequence.
# - Set `learning_rate` to the learning rate.
# - Set `show_every_n_batches` to the number of batches the neural network should print progress.

# In[15]:


# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 1024
# Embedding Dimension Size
embed_dim = 256
# Sequence Length
seq_length = 15
# Learning Rate
learning_rate = 0.003
# Show stats for every n number of batches
show_every_n_batches = 10

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'


# ### Build the Graph
# ### 构建图
# Build the graph using the neural network you implemented.
# 
# 使用您实施的神经网络来构建图。

# In[16]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    keep_prob = get_keep_prob()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size, keep_prob)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


# ## Train
# ## 训练
# Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.
# 
# 在预处理的数据上训练神经网络。 如果您很难获得良好的损失，请查看[forms](https://discussions.udacity.com/)，看看是否有人遇到同样的问题。
# 

# In[17]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate,
                keep_prob:0.7}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# ## Save Parameters
# ## 保存参数
# Save `seq_length` and `save_dir` for generating a new TV script.
# 
# 保存`seq_length`和`save_dir`来生成一个新的TV脚本。
# 

# In[18]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))


# # Checkpoint
# # 检查点

# In[19]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()


# ## Implement Generate Functions
# ## 实现生成函数
# ### Get Tensors
# ### 获取Tensors
# Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
# 
# 使用函数[`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name)从`loaded_graph`获取张量。 使用以下名称获取张量：
# 
# - "input:0"
# - "initial_state:0"
# - "final_state:0"
# - "probs:0"
# 
# Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 
# 
# 返回以下元组中的张量`（InputTensor，InitialStateTensor，FinalStateTensor，ProbsTensor）`

# In[20]:


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    input_tensor = loaded_graph.get_tensor_by_name('input:0')
    init_state_tensor = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state_tensor = loaded_graph.get_tensor_by_name('final_state:0')
    probs_tensor = loaded_graph.get_tensor_by_name('probs:0')
    
    return input_tensor, init_state_tensor, final_state_tensor, probs_tensor


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)


# In[21]:


def get_keep_porob_tensors(loaded_graph):
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    return keep_prob


# ### Choose Word
# ### 选择单词
# 
# Implement the `pick_word()` function to select the next word using `probabilities`.
# 
# 实现`pick_word（）`函数来使用`probabilities`选择下一个单词。

# In[22]:


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    idx = np.argmax(probabilities)
    return int_to_vocab[idx]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)


# ## Generate TV Script
# ## 生成电视 剧本
# This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.
# 
# 这将为您生成电视剧。 将`gen_length`设置为您想要生成的电视剧的长度。

# In[23]:


gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)
    keep_prob = get_keep_porob_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state, keep_prob:1})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)


# # The TV Script is Nonsensical
# # 电视剧本是无意义的
# 
# It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# 
# 如果电视剧本没有任何意义的话，那也没关系。 我们训练了不到一兆字节的文本。 为了获得好的结果，你将不得不使用更小的词汇量或获得更多的数据。 幸运的是，还有更多的数据！ 正如我们在这个项目讨论中提到的那样，这是[另一个数据集](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data)的一个子集。 我们没有训练所有的数据，因为这将花费太长时间。 但是，您可以自由地在所有数据上训练您的神经网络。 当然，在你完成这个项目之后。...
# 
# # Submitting This Project
# # 提交项目
# 
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
# 
# 提交此项目时，请确保在保存笔记本之前运行所有单元。 将笔记本文件保存为“dlnd_tv_script_generation.ipynb”，并将其保存为“文件” - >“下载为”下的HTML文件。 在提交中包含“helper.py”和“problem_unittests.py”文件。

# ## 问题
# 1. lstm的zero_state与batch_size有什么关系，是因为lstm中用于长时记忆的单元状态的size与batch_size有关么？
# 2. embedding的时候为什么大部分资料都使用random_normal，而不用tf.truncated_normal,stddev=0.1来初始化权重？
# 3. 为什么单层RNN网络效果比多层的RNN网络效果好，什么情况下适合使用多层的RNN网络？
# 4. 疑惑很久的问题，就该项目描述下我对于RNN网络的理解，假设一个RNN网络的物理结构有1个embedding层、3个LSTM层（n个cell）和最后的1个全连接层，每次的输入是一个单词，用于预测下一个单词，就是说只有在当前单词输入之后才能输入下一个单词，否则网络等于没有学习当前单词及其序列关系，在对RNN网络按时间展开后，每次可以输入一个seq_length长度个单词，输出另外一个seq_length长度的单词，在展开后的网络中，实际应该有并行的seq_length个RNN网络，也就有seq_length个全连接层（每次时间点一个），那么为什么在项目中dynamic_rnn之后只建立了一个全连接层，不是seq_length个全连接层，而输入全连接层的数据是包含批次和序列信息的dynamic_rnn的所有输出，这里太容易让人混淆了。另外， 输入到全连接层的数据是每个时间点输入一次的还是每个sequence一起进行一次输入，就是说在RNN输出后，所输入的数据是一起输入给全连接层还是按照时间顺序依次输入给全连接层？如果是按时间序列依次输入给全连接层，为什么在Anna KaRNNa项目中，要对RNN的输出进行concat和reshape后利用tf.matmul计算logits，这里在reshape之后，张量不就变成2维的了么，最后不就只能有一个字母的预测了么，这样不是丢失了序列信息了么?

# lstm_layers = 1
#  
# num_epochs = 100
# 
# batch_size = 128
# 
# rnn_size = 512
# 
# embed_dim = 256
# 
# seq_length = 25
# 
# learning_rate = 0.002
# 
# show_every_n_batches = 10
# 
# - Epoch   0 Batch    0/21   train_loss = 8.826
# - Epoch   0 Batch   10/21   train_loss = 6.155
# - Epoch   0 Batch   20/21   train_loss = 5.792
# - Epoch   1 Batch    9/21   train_loss = 5.363
# - Epoch   1 Batch   19/21   train_loss = 5.112
# - Epoch   2 Batch    8/21   train_loss = 4.848
# - Epoch   2 Batch   18/21   train_loss = 4.717
# - Epoch   3 Batch    7/21   train_loss = 4.503
# - Epoch   3 Batch   17/21   train_loss = 4.426
# - Epoch   4 Batch    6/21   train_loss = 4.311
# - Epoch   4 Batch   16/21   train_loss = 4.225
# - Epoch   5 Batch    5/21   train_loss = 3.997
# - Epoch   5 Batch   15/21   train_loss = 3.895
# - Epoch   6 Batch    4/21   train_loss = 3.877
# - Epoch   6 Batch   14/21   train_loss = 3.598
# - Epoch   7 Batch    3/21   train_loss = 3.587
# - Epoch   7 Batch   13/21   train_loss = 3.441
# - Epoch   8 Batch    2/21   train_loss = 3.424
# - Epoch   8 Batch   12/21   train_loss = 3.212
# - Epoch   9 Batch    1/21   train_loss = 3.093
# - Epoch   9 Batch   11/21   train_loss = 2.976
# - Epoch  10 Batch    0/21   train_loss = 2.948
# - Epoch  10 Batch   10/21   train_loss = 2.799
# - Epoch  10 Batch   20/21   train_loss = 2.851
# - Epoch  11 Batch    9/21   train_loss = 2.652
# - Epoch  11 Batch   19/21   train_loss = 2.557
# - Epoch  12 Batch    8/21   train_loss = 2.587
# - Epoch  12 Batch   18/21   train_loss = 2.466
# - Epoch  13 Batch    7/21   train_loss = 2.422
# - Epoch  13 Batch   17/21   train_loss = 2.288
# - Epoch  14 Batch    6/21   train_loss = 2.264
# - Epoch  14 Batch   16/21   train_loss = 2.189
# - Epoch  15 Batch    5/21   train_loss = 2.186
# - Epoch  15 Batch   15/21   train_loss = 2.116
# - Epoch  16 Batch    4/21   train_loss = 2.048
# - Epoch  16 Batch   14/21   train_loss = 1.926
# - Epoch  17 Batch    3/21   train_loss = 1.914
# - Epoch  17 Batch   13/21   train_loss = 1.920
# - Epoch  18 Batch    2/21   train_loss = 1.937
# - Epoch  18 Batch   12/21   train_loss = 1.827
# - Epoch  19 Batch    1/21   train_loss = 1.764
# - Epoch  19 Batch   11/21   train_loss = 1.764
# - Epoch  20 Batch    0/21   train_loss = 1.749
# - Epoch  20 Batch   10/21   train_loss = 1.660
# - Epoch  20 Batch   20/21   train_loss = 1.704
# - Epoch  21 Batch    9/21   train_loss = 1.560
# - Epoch  21 Batch   19/21   train_loss = 1.488
# - Epoch  22 Batch    8/21   train_loss = 1.515
# - Epoch  22 Batch   18/21   train_loss = 1.504
# - Epoch  23 Batch    7/21   train_loss = 1.462
# - Epoch  23 Batch   17/21   train_loss = 1.365
# - Epoch  24 Batch    6/21   train_loss = 1.352
# - Epoch  24 Batch   16/21   train_loss = 1.247
# - Epoch  25 Batch    5/21   train_loss = 1.283
# - Epoch  25 Batch   15/21   train_loss = 1.265
# - Epoch  26 Batch    4/21   train_loss = 1.148
# - Epoch  26 Batch   14/21   train_loss = 1.135
# - Epoch  27 Batch    3/21   train_loss = 1.125
# - Epoch  27 Batch   13/21   train_loss = 1.102
# - Epoch  28 Batch    2/21   train_loss = 1.112
# - Epoch  28 Batch   12/21   train_loss = 0.989
# - Epoch  29 Batch    1/21   train_loss = 0.958
# - Epoch  29 Batch   11/21   train_loss = 0.997
# - Epoch  30 Batch    0/21   train_loss = 0.970
# - Epoch  30 Batch   10/21   train_loss = 0.948
# - Epoch  30 Batch   20/21   train_loss = 0.923
# - Epoch  31 Batch    9/21   train_loss = 0.880
# - Epoch  31 Batch   19/21   train_loss = 0.833
# - Epoch  32 Batch    8/21   train_loss = 0.894
# - Epoch  32 Batch   18/21   train_loss = 0.856
# - Epoch  33 Batch    7/21   train_loss = 0.784
# - Epoch  33 Batch   17/21   train_loss = 0.826
# - Epoch  34 Batch    6/21   train_loss = 0.789
# - Epoch  34 Batch   16/21   train_loss = 0.736
# - Epoch  35 Batch    5/21   train_loss = 0.731
# - Epoch  35 Batch   15/21   train_loss = 0.722
# - Epoch  36 Batch    4/21   train_loss = 0.651
# - Epoch  36 Batch   14/21   train_loss = 0.654
# - Epoch  37 Batch    3/21   train_loss = 0.640
# - Epoch  37 Batch   13/21   train_loss = 0.633
# - Epoch  38 Batch    2/21   train_loss = 0.653
# - Epoch  38 Batch   12/21   train_loss = 0.590
# - Epoch  39 Batch    1/21   train_loss = 0.576
# - Epoch  39 Batch   11/21   train_loss = 0.594
# - Epoch  40 Batch    0/21   train_loss = 0.599
# - Epoch  40 Batch   10/21   train_loss = 0.538
# - Epoch  40 Batch   20/21   train_loss = 0.526
# - Epoch  41 Batch    9/21   train_loss = 0.514
# - Epoch  41 Batch   19/21   train_loss = 0.509
# - Epoch  42 Batch    8/21   train_loss = 0.538
# - Epoch  42 Batch   18/21   train_loss = 0.512
# - Epoch  43 Batch    7/21   train_loss = 0.539
# - Epoch  43 Batch   17/21   train_loss = 0.502
# - Epoch  44 Batch    6/21   train_loss = 0.474
# - Epoch  44 Batch   16/21   train_loss = 0.456
# - Epoch  45 Batch    5/21   train_loss = 0.457
# - Epoch  45 Batch   15/21   train_loss = 0.438
# - Epoch  46 Batch    4/21   train_loss = 0.413
# - Epoch  46 Batch   14/21   train_loss = 0.392
# - Epoch  47 Batch    3/21   train_loss = 0.411
# - Epoch  47 Batch   13/21   train_loss = 0.402
# - Epoch  48 Batch    2/21   train_loss = 0.433
# - Epoch  48 Batch   12/21   train_loss = 0.368
# - Epoch  49 Batch    1/21   train_loss = 0.361
# - Epoch  49 Batch   11/21   train_loss = 0.383
# - Epoch  50 Batch    0/21   train_loss = 0.399
# - Epoch  50 Batch   10/21   train_loss = 0.347
# - Epoch  50 Batch   20/21   train_loss = 0.363
# - Epoch  51 Batch    9/21   train_loss = 0.358
# - Epoch  51 Batch   19/21   train_loss = 0.350
# - Epoch  52 Batch    8/21   train_loss = 0.375
# - Epoch  52 Batch   18/21   train_loss = 0.337
# - Epoch  53 Batch    7/21   train_loss = 0.340
# - Epoch  53 Batch   17/21   train_loss = 0.354
# - Epoch  54 Batch    6/21   train_loss = 0.349
# - Epoch  54 Batch   16/21   train_loss = 0.319
# - Epoch  55 Batch    5/21   train_loss = 0.331
# - Epoch  55 Batch   15/21   train_loss = 0.329
# - Epoch  56 Batch    4/21   train_loss = 0.306
# - Epoch  56 Batch   14/21   train_loss = 0.283
# - Epoch  57 Batch    3/21   train_loss = 0.319
# - Epoch  57 Batch   13/21   train_loss = 0.290
# - Epoch  58 Batch    2/21   train_loss = 0.329
# - Epoch  58 Batch   12/21   train_loss = 0.285
# - Epoch  59 Batch    1/21   train_loss = 0.290
# - Epoch  59 Batch   11/21   train_loss = 0.285
# - Epoch  60 Batch    0/21   train_loss = 0.274
# - Epoch  60 Batch   10/21   train_loss = 0.240
# - Epoch  60 Batch   20/21   train_loss = 0.264
# - Epoch  61 Batch    9/21   train_loss = 0.276
# - Epoch  61 Batch   19/21   train_loss = 0.272
# - Epoch  62 Batch    8/21   train_loss = 0.277
# - Epoch  62 Batch   18/21   train_loss = 0.266
# - Epoch  63 Batch    7/21   train_loss = 0.259
# - Epoch  63 Batch   17/21   train_loss = 0.253
# - Epoch  64 Batch    6/21   train_loss = 0.243
# - Epoch  64 Batch   16/21   train_loss = 0.250
# - Epoch  65 Batch    5/21   train_loss = 0.246
# - Epoch  65 Batch   15/21   train_loss = 0.247
# - Epoch  66 Batch    4/21   train_loss = 0.231
# - Epoch  66 Batch   14/21   train_loss = 0.231
# - Epoch  67 Batch    3/21   train_loss = 0.247
# - Epoch  67 Batch   13/21   train_loss = 0.229
# - Epoch  68 Batch    2/21   train_loss = 0.252
# - Epoch  68 Batch   12/21   train_loss = 0.217
# - Epoch  69 Batch    1/21   train_loss = 0.222
# - Epoch  69 Batch   11/21   train_loss = 0.227
# - Epoch  70 Batch    0/21   train_loss = 0.226
# - Epoch  70 Batch   10/21   train_loss = 0.205
# - Epoch  70 Batch   20/21   train_loss = 0.225
# - Epoch  71 Batch    9/21   train_loss = 0.236
# - Epoch  71 Batch   19/21   train_loss = 0.236
# - Epoch  72 Batch    8/21   train_loss = 0.234
# - Epoch  72 Batch   18/21   train_loss = 0.240
# - Epoch  73 Batch    7/21   train_loss = 0.216
# - Epoch  73 Batch   17/21   train_loss = 0.214
# - Epoch  74 Batch    6/21   train_loss = 0.219
# - Epoch  74 Batch   16/21   train_loss = 0.217
# - Epoch  75 Batch    5/21   train_loss = 0.228
# - Epoch  75 Batch   15/21   train_loss = 0.212
# - Epoch  76 Batch    4/21   train_loss = 0.239
# - Epoch  76 Batch   14/21   train_loss = 0.207
# - Epoch  77 Batch    3/21   train_loss = 0.226
# - Epoch  77 Batch   13/21   train_loss = 0.200
# - Epoch  78 Batch    2/21   train_loss = 0.234
# - Epoch  78 Batch   12/21   train_loss = 0.194
# - Epoch  79 Batch    1/21   train_loss = 0.206
# - Epoch  79 Batch   11/21   train_loss = 0.210
# - Epoch  80 Batch    0/21   train_loss = 0.230
# - Epoch  80 Batch   10/21   train_loss = 0.196
# - Epoch  80 Batch   20/21   train_loss = 0.192
# - Epoch  81 Batch    9/21   train_loss = 0.210
# - Epoch  81 Batch   19/21   train_loss = 0.199
# - Epoch  82 Batch    8/21   train_loss = 0.207
# - Epoch  82 Batch   18/21   train_loss = 0.201
# - Epoch  83 Batch    7/21   train_loss = 0.203
# - Epoch  83 Batch   17/21   train_loss = 0.195
# - Epoch  84 Batch    6/21   train_loss = 0.219
# - Epoch  84 Batch   16/21   train_loss = 0.203
# - Epoch  85 Batch    5/21   train_loss = 0.195
# - Epoch  85 Batch   15/21   train_loss = 0.208
# - Epoch  86 Batch    4/21   train_loss = 0.196
# - Epoch  86 Batch   14/21   train_loss = 0.183
# - Epoch  87 Batch    3/21   train_loss = 0.204
# - Epoch  87 Batch   13/21   train_loss = 0.192
# - Epoch  88 Batch    2/21   train_loss = 0.214
# - Epoch  88 Batch   12/21   train_loss = 0.195
# - Epoch  89 Batch    1/21   train_loss = 0.205
# - Epoch  89 Batch   11/21   train_loss = 0.198
# - Epoch  90 Batch    0/21   train_loss = 0.206
# - Epoch  90 Batch   10/21   train_loss = 0.182
# - Epoch  90 Batch   20/21   train_loss = 0.192
# - Epoch  91 Batch    9/21   train_loss = 0.204
# - Epoch  91 Batch   19/21   train_loss = 0.176
# - Epoch  92 Batch    8/21   train_loss = 0.199
# - Epoch  92 Batch   18/21   train_loss = 0.197
# - Epoch  93 Batch    7/21   train_loss = 0.195
# - Epoch  93 Batch   17/21   train_loss = 0.186
# - Epoch  94 Batch    6/21   train_loss = 0.181
# - Epoch  94 Batch   16/21   train_loss = 0.190
# - Epoch  95 Batch    5/21   train_loss = 0.188
# - Epoch  95 Batch   15/21   train_loss = 0.181
# - Epoch  96 Batch    4/21   train_loss = 0.186
# - Epoch  96 Batch   14/21   train_loss = 0.179
# - Epoch  97 Batch    3/21   train_loss = 0.207
# - Epoch  97 Batch   13/21   train_loss = 0.189
# - Epoch  98 Batch    2/21   train_loss = 0.220
# - Epoch  98 Batch   12/21   train_loss = 0.187
# - Epoch  99 Batch    1/21   train_loss = 0.200
# - Epoch  99 Batch   11/21   train_loss = 0.194
# - Model Trained and Saved
# 
# 
# moe_szyslak:(looking at homer) you're right. he needs some professional help...
# duffman: ooh, someone is down in the sack
# homer_simpson: i can't believe you don't be your foot, homer.
# homer_simpson: i had the greatest gift of all, a little girl who could pick me to go halvsies on a ring.
# edna_krabappel-flanders: seymour...(ad lib singing) my bar could be in.
# homer_simpson:(derisive snort) kent brockman!
# homer_simpson:(touched) aw, that's my fourth grade teacher!
# carl_carlson: are you gonna be okay?
# barney_gumble:(reciting)" your infatuation is based on a physical attraction. talk to the woman of a way to paris...
# homer_simpson:(chuckles at injury) yeah, but at least we're hearing some interesting conversation from those two book clubs.
# book_club_member: well, well, look who's it.
# moe_szyslak:(amid men's reactions) you got that right!
# seymour_skinner: edna won't even let me in here?
# moe_szyslak:(nods) keep my tail

# lstm_layers = 1
#  
# 
# num_epochs = 100
# 
# batch_size = 128
# 
# rnn_size = 512
# 
# embed_dim = 256
# 
# seq_length = 25
# 
# learning_rate = 0.005
# 
# show_every_n_batches = 10
# 
# - Epoch   0 Batch    0/21   train_loss = 8.821
# - Epoch   0 Batch   10/21   train_loss = 6.080
# - Epoch   0 Batch   20/21   train_loss = 5.732
# - Epoch   1 Batch    9/21   train_loss = 5.324
# - Epoch   1 Batch   19/21   train_loss = 5.043
# - Epoch   2 Batch    8/21   train_loss = 4.830
# - Epoch   2 Batch   18/21   train_loss = 4.676
# - Epoch   3 Batch    7/21   train_loss = 4.487
# - Epoch   3 Batch   17/21   train_loss = 4.379
# - Epoch   4 Batch    6/21   train_loss = 4.250
# - Epoch   4 Batch   16/21   train_loss = 4.130
# - Epoch   5 Batch    5/21   train_loss = 3.969
# - Epoch   5 Batch   15/21   train_loss = 3.855
# - Epoch   6 Batch    4/21   train_loss = 3.836
# - Epoch   6 Batch   14/21   train_loss = 3.543
# - Epoch   7 Batch    3/21   train_loss = 3.556
# - Epoch   7 Batch   13/21   train_loss = 3.392
# - Epoch   8 Batch    2/21   train_loss = 3.375
# - Epoch   8 Batch   12/21   train_loss = 3.161
# - Epoch   9 Batch    1/21   train_loss = 3.066
# - Epoch   9 Batch   11/21   train_loss = 2.921
# - Epoch  10 Batch    0/21   train_loss = 2.914
# - Epoch  10 Batch   10/21   train_loss = 2.747
# - Epoch  10 Batch   20/21   train_loss = 2.785
# - Epoch  11 Batch    9/21   train_loss = 2.561
# - Epoch  11 Batch   19/21   train_loss = 2.517
# - Epoch  12 Batch    8/21   train_loss = 2.472
# - Epoch  12 Batch   18/21   train_loss = 2.465
# - Epoch  13 Batch    7/21   train_loss = 2.287
# - Epoch  13 Batch   17/21   train_loss = 2.204
# - Epoch  14 Batch    6/21   train_loss = 2.219
# - Epoch  14 Batch   16/21   train_loss = 2.129
# - Epoch  15 Batch    5/21   train_loss = 2.144
# - Epoch  15 Batch   15/21   train_loss = 2.043
# - Epoch  16 Batch    4/21   train_loss = 1.970
# - Epoch  16 Batch   14/21   train_loss = 1.869
# - Epoch  17 Batch    3/21   train_loss = 1.856
# - Epoch  17 Batch   13/21   train_loss = 1.842
# - Epoch  18 Batch    2/21   train_loss = 1.911
# - Epoch  18 Batch   12/21   train_loss = 1.711
# - Epoch  19 Batch    1/21   train_loss = 1.657
# - Epoch  19 Batch   11/21   train_loss = 1.650
# - Epoch  20 Batch    0/21   train_loss = 1.666
# - Epoch  20 Batch   10/21   train_loss = 1.575
# - Epoch  20 Batch   20/21   train_loss = 1.571
# - Epoch  21 Batch    9/21   train_loss = 1.490
# - Epoch  21 Batch   19/21   train_loss = 1.444
# - Epoch  22 Batch    8/21   train_loss = 1.440
# - Epoch  22 Batch   18/21   train_loss = 1.395
# - Epoch  23 Batch    7/21   train_loss = 1.361
# - Epoch  23 Batch   17/21   train_loss = 1.257
# - Epoch  24 Batch    6/21   train_loss = 1.269
# - Epoch  24 Batch   16/21   train_loss = 1.220
# - Epoch  25 Batch    5/21   train_loss = 1.271
# - Epoch  25 Batch   15/21   train_loss = 1.198
# - Epoch  26 Batch    4/21   train_loss = 1.149
# - Epoch  26 Batch   14/21   train_loss = 1.095
# - Epoch  27 Batch    3/21   train_loss = 1.073
# - Epoch  27 Batch   13/21   train_loss = 1.073
# - Epoch  28 Batch    2/21   train_loss = 1.104
# - Epoch  28 Batch   12/21   train_loss = 1.032
# - Epoch  29 Batch    1/21   train_loss = 0.980
# - Epoch  29 Batch   11/21   train_loss = 1.006
# - Epoch  30 Batch    0/21   train_loss = 0.974
# - Epoch  30 Batch   10/21   train_loss = 0.959
# - Epoch  30 Batch   20/21   train_loss = 0.975
# - Epoch  31 Batch    9/21   train_loss = 0.930
# - Epoch  31 Batch   19/21   train_loss = 0.915
# - Epoch  32 Batch    8/21   train_loss = 0.935
# - Epoch  32 Batch   18/21   train_loss = 0.916
# - Epoch  33 Batch    7/21   train_loss = 0.838
# - Epoch  33 Batch   17/21   train_loss = 0.845
# - Epoch  34 Batch    6/21   train_loss = 0.792
# - Epoch  34 Batch   16/21   train_loss = 0.771
# - Epoch  35 Batch    5/21   train_loss = 0.800
# - Epoch  35 Batch   15/21   train_loss = 0.770
# - Epoch  36 Batch    4/21   train_loss = 0.692
# - Epoch  36 Batch   14/21   train_loss = 0.688
# - Epoch  37 Batch    3/21   train_loss = 0.674
# - Epoch  37 Batch   13/21   train_loss = 0.640
# - Epoch  38 Batch    2/21   train_loss = 0.694
# - Epoch  38 Batch   12/21   train_loss = 0.615
# - Epoch  39 Batch    1/21   train_loss = 0.592
# - Epoch  39 Batch   11/21   train_loss = 0.601
# - Epoch  40 Batch    0/21   train_loss = 0.602
# - Epoch  40 Batch   10/21   train_loss = 0.559
# - Epoch  40 Batch   20/21   train_loss = 0.542
# - Epoch  41 Batch    9/21   train_loss = 0.529
# - Epoch  41 Batch   19/21   train_loss = 0.502
# - Epoch  42 Batch    8/21   train_loss = 0.499
# - Epoch  42 Batch   18/21   train_loss = 0.510
# - Epoch  43 Batch    7/21   train_loss = 0.489
# - Epoch  43 Batch   17/21   train_loss = 0.459
# - Epoch  44 Batch    6/21   train_loss = 0.461
# - Epoch  44 Batch   16/21   train_loss = 0.459
# - Epoch  45 Batch    5/21   train_loss = 0.459
# - Epoch  45 Batch   15/21   train_loss = 0.441
# - Epoch  46 Batch    4/21   train_loss = 0.408
# - Epoch  46 Batch   14/21   train_loss = 0.384
# - Epoch  47 Batch    3/21   train_loss = 0.416
# - Epoch  47 Batch   13/21   train_loss = 0.393
# - Epoch  48 Batch    2/21   train_loss = 0.435
# - Epoch  48 Batch   12/21   train_loss = 0.402
# - Epoch  49 Batch    1/21   train_loss = 0.382
# - Epoch  49 Batch   11/21   train_loss = 0.401
# - Epoch  50 Batch    0/21   train_loss = 0.391
# - Epoch  50 Batch   10/21   train_loss = 0.354
# - Epoch  50 Batch   20/21   train_loss = 0.357
# - Epoch  51 Batch    9/21   train_loss = 0.365
# - Epoch  51 Batch   19/21   train_loss = 0.361
# - Epoch  52 Batch    8/21   train_loss = 0.354
# - Epoch  52 Batch   18/21   train_loss = 0.331
# - Epoch  53 Batch    7/21   train_loss = 0.335
# - Epoch  53 Batch   17/21   train_loss = 0.313
# - Epoch  54 Batch    6/21   train_loss = 0.334
# - Epoch  54 Batch   16/21   train_loss = 0.325
# - Epoch  55 Batch    5/21   train_loss = 0.332
# - Epoch  55 Batch   15/21   train_loss = 0.310
# - Epoch  56 Batch    4/21   train_loss = 0.329
# - Epoch  56 Batch   14/21   train_loss = 0.283
# - Epoch  57 Batch    3/21   train_loss = 0.299
# - Epoch  57 Batch   13/21   train_loss = 0.287
# - Epoch  58 Batch    2/21   train_loss = 0.307
# - Epoch  58 Batch   12/21   train_loss = 0.284
# - Epoch  59 Batch    1/21   train_loss = 0.294
# - Epoch  59 Batch   11/21   train_loss = 0.290
# - Epoch  60 Batch    0/21   train_loss = 0.298
# - Epoch  60 Batch   10/21   train_loss = 0.259
# - Epoch  60 Batch   20/21   train_loss = 0.277
# - Epoch  61 Batch    9/21   train_loss = 0.283
# - Epoch  61 Batch   19/21   train_loss = 0.263
# - Epoch  62 Batch    8/21   train_loss = 0.271
# - Epoch  62 Batch   18/21   train_loss = 0.265
# - Epoch  63 Batch    7/21   train_loss = 0.272
# - Epoch  63 Batch   17/21   train_loss = 0.254
# - Epoch  64 Batch    6/21   train_loss = 0.261
# - Epoch  64 Batch   16/21   train_loss = 0.264
# - Epoch  65 Batch    5/21   train_loss = 0.271
# - Epoch  65 Batch   15/21   train_loss = 0.255
# - Epoch  66 Batch    4/21   train_loss = 0.251
# - Epoch  66 Batch   14/21   train_loss = 0.246
# - Epoch  67 Batch    3/21   train_loss = 0.253
# - Epoch  67 Batch   13/21   train_loss = 0.234
# - Epoch  68 Batch    2/21   train_loss = 0.258
# - Epoch  68 Batch   12/21   train_loss = 0.226
# - Epoch  69 Batch    1/21   train_loss = 0.237
# - Epoch  69 Batch   11/21   train_loss = 0.244
# - Epoch  70 Batch    0/21   train_loss = 0.254
# - Epoch  70 Batch   10/21   train_loss = 0.222
# - Epoch  70 Batch   20/21   train_loss = 0.221
# - Epoch  71 Batch    9/21   train_loss = 0.234
# - Epoch  71 Batch   19/21   train_loss = 0.234
# - Epoch  72 Batch    8/21   train_loss = 0.230
# - Epoch  72 Batch   18/21   train_loss = 0.235
# - Epoch  73 Batch    7/21   train_loss = 0.229
# - Epoch  73 Batch   17/21   train_loss = 0.215
# - Epoch  74 Batch    6/21   train_loss = 0.231
# - Epoch  74 Batch   16/21   train_loss = 0.223
# - Epoch  75 Batch    5/21   train_loss = 0.224
# - Epoch  75 Batch   15/21   train_loss = 0.227
# - Epoch  76 Batch    4/21   train_loss = 0.216
# - Epoch  76 Batch   14/21   train_loss = 0.201
# - Epoch  77 Batch    3/21   train_loss = 0.210
# - Epoch  77 Batch   13/21   train_loss = 0.206
# - Epoch  78 Batch    2/21   train_loss = 0.245
# - Epoch  78 Batch   12/21   train_loss = 0.211
# - Epoch  79 Batch    1/21   train_loss = 0.210
# - Epoch  79 Batch   11/21   train_loss = 0.227
# - Epoch  80 Batch    0/21   train_loss = 0.223
# - Epoch  80 Batch   10/21   train_loss = 0.200
# - Epoch  80 Batch   20/21   train_loss = 0.212
# - Epoch  81 Batch    9/21   train_loss = 0.220
# - Epoch  81 Batch   19/21   train_loss = 0.194
# - Epoch  82 Batch    8/21   train_loss = 0.221
# - Epoch  82 Batch   18/21   train_loss = 0.216
# - Epoch  83 Batch    7/21   train_loss = 0.212
# - Epoch  83 Batch   17/21   train_loss = 0.228
# - Epoch  84 Batch    6/21   train_loss = 0.219
# - Epoch  84 Batch   16/21   train_loss = 0.213
# - Epoch  85 Batch    5/21   train_loss = 0.225
# - Epoch  85 Batch   15/21   train_loss = 0.221
# - Epoch  86 Batch    4/21   train_loss = 0.217
# - Epoch  86 Batch   14/21   train_loss = 0.202
# - Epoch  87 Batch    3/21   train_loss = 0.226
# - Epoch  87 Batch   13/21   train_loss = 0.210
# - Epoch  88 Batch    2/21   train_loss = 0.243
# - Epoch  88 Batch   12/21   train_loss = 0.206
# - Epoch  89 Batch    1/21   train_loss = 0.222
# - Epoch  89 Batch   11/21   train_loss = 0.230
# - Epoch  90 Batch    0/21   train_loss = 0.234
# - Epoch  90 Batch   10/21   train_loss = 0.212
# - Epoch  90 Batch   20/21   train_loss = 0.210
# - Epoch  91 Batch    9/21   train_loss = 0.242
# - Epoch  91 Batch   19/21   train_loss = 0.219
# - Epoch  92 Batch    8/21   train_loss = 0.228
# - Epoch  92 Batch   18/21   train_loss = 0.225
# - Epoch  93 Batch    7/21   train_loss = 0.222
# - Epoch  93 Batch   17/21   train_loss = 0.203
# - Epoch  94 Batch    6/21   train_loss = 0.223
# - Epoch  94 Batch   16/21   train_loss = 0.215
# - Epoch  95 Batch    5/21   train_loss = 0.224
# - Epoch  95 Batch   15/21   train_loss = 0.218
# - Epoch  96 Batch    4/21   train_loss = 0.226
# - Epoch  96 Batch   14/21   train_loss = 0.209
# - Epoch  97 Batch    3/21   train_loss = 0.243
# - Epoch  97 Batch   13/21   train_loss = 0.218
# - Epoch  98 Batch    2/21   train_loss = 0.251
# - Epoch  98 Batch   12/21   train_loss = 0.217
# - Epoch  99 Batch    1/21   train_loss = 0.231
# - Epoch  99 Batch   11/21   train_loss = 0.236
# - Model Trained and Saved
# 
# moe_szyslak:(uneasy) oh, i can't wait till you guys get to uh...
# thought_bubble_lenny: yep, that's what we'd do, moe.
# moe_szyslak:(snorts) nobody does.
# kemi:(yawns) i haven't eaten all day.
# moe_szyslak: don't eat those eggs, homer.
# homer_simpson: you came to the right guy. i'll straighten ya out...(nervous laugh)
# moe_szyslak:(sniffles, then, impatient) if they was me!
# homer_simpson:(angrily) moe, what are you doing?
# homer_simpson:(slyly) but i think does this the springfield lottery.
# homer_simpson:(sighs) well, i aims to please. hey,... uh...
# agent_miller: homer simpson?(flashing badge)
# homer_simpson:(to scully) you are one at me!
# moe_szyslak: now, you know i can't sell you no beer till two knives.
# lenny_leonard:(mid-conversation) so who do you like, the padres or is it?
# moe_szyslak:(re: homer) hey, i
# 

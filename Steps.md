# LLM from Scratch
- A decoder-only LLM
## Arch Steps
![alt text](image.png)

## Preparing Dataset
### Tokenizations
- Splitting words to tokens
- Involves creating text sampling
- Covert token into vectors - called Embeddings
- Can embed different formats - text,video,audio
- Embedding are done by embed models for each specific data type

![alt text](image-1.png)

### Embedding defined
- Is a mapping from discrete objects e.g words, images, documents etc to a POINTS in a continous vector space. 
- This is key in converting non-numerical data into a format that neural networks can understand.
- Embeddings can be words, sentences etc

In this project we will focus on word embedings using Word2Vec

### 2D Word2Vec
- Word2Vec is a pre-trained model for word embeddings generations
- Word embeddings can have 1 to 1000s of dimensions.
- GPT-2 models with 117million and 125million params uses an embedding size of 768 deimesions
- Largest GPT-3 model with 175b params uses a embedding of 12,288 dimesions
- Higher dimensionality captures nuanced relationships but at high cost of computation

### Note
*LLMs* produce their own embeddings that r part of input and are updated during training - this ensures embeddings are updated to specific task and data at hand.


#### Sample scatter plot for a 2D word embendings
![alt text](image-2.png)


### Tokenization
- We will use [The verdict] story text - this is on wiki.
#### Decoder-only LLM tokenizations steps
![alt text](image-3.png)

#### Creating token IDs
- token ID is an integer number assigned to each unique token. Forming a vocabulary matrix

![alt text](image-4.png)

#### Tokenizer.py 
- Below is the implementation and how it is used:
![alt text](image-5.png)

#### handling unrelated text sources with special context tokens
- unk - unknown words - "<|unk|>"
- endoftext - "<|endoftext|>"
- BOS - beginning of sequence - mark start of text
- EOS - end of sequence - similar ot endoftext
- PAD - padding - extending texts to match/unify lengths in training data
- 

![alt text](image-6.png)

New tokenizer version 2 illustrates use of special context token unk

GPT models do not use unk but rather byte pair encoding BPE

#### BPE
Byte pair encoding - the tokenizer used in training GPT 3
We will use tiktoken library
- BPE handles out of context words such as someunkownPlace by splitting them into subwords or even chars.

![img.png](img.png)

### Data sampling - sliding window
- prediction process
- using tiktoken BPE encoder
- using pytorch data loader
#### Visuals
![img_1.png](img_1.png)

#### Data loading - pytorch
- Pytorch tensors - multidimensional arrays of LLM inputs(text that LLM sees) & targets(what LLM should predict)

![img_2.png](img_2.png)

![img_3.png](img_3.png)

#### Key variables in pytorch sensors-based data loading
- max size = the number of inputs in each tensor - LLM trains on input size = 256
- stride = the number of positions the input shift across the batches (sliding window), 
if stride = max_size/input-size then there will be no intersection between batches
![img_4.png](img_4.png)

- Batch Size - the number of pairs of tensors to load. batch size of 1 requires less momory but noisy model updates
it is there4 useful for illustrations but not actual training. We can use batch size > 1 and below is the output:

![img_5.png](img_5.png)
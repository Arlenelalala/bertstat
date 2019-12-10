# Bertstat

This is a BERT parameters analyzer based on [torchstat](https://github.com/Swall0w/torchstat).  

This tools can show：

- Total number of BERT parameters
- BERT FLOPs
- BERT MAdd
- BERT Memory usage
- Histogram and Box plot of BERT parameters



# Usage

* git clone https://github.com/Arlenelalala/Bertstat

```python
from Bertstat import bert_stat
bert_stat(bert_path, input_size=(1,1))
```

* bert_path: your BERT(pytorch version) path
* input_size: Tuple(batch_size, seq_length)



# BERT 

![](https://i.imgur.com/PuKEb5T.jpg) 

l_max: the max sequence length, 512

vocab: the number of vocabulary, 30522

L ：the number of layers 

H：the hidden size , $H=d_{model}=d_k∗A$

A ：the number of self-attention heads

d_1：feed-forward size

* **Embedding layers**

  ![](https://i.imgur.com/6Zuk9fY.jpg)

* **Encoder layers**

  Encoder layers has L Transformer layer. There is a Transformer encoder layer.

  ![](https://i.imgur.com/Cuef0q8.jpg)

* **Pooler layers**

  Pooler layers is an extra linear projection layer which weigh is $d_{model} * d_{model}$ and bias is $1*d_{model}$. 
  And the number of pooler layers parameter is $d_{model}∗(d_{model}+1)$.

* Parameter number and MAdd

    | Model                                                        |                                    | Para Num  | MAdd                     |
  | ------------------------------------------------------------ | ---------------------------------- | --------- | ------------------------ |
  | **[BERT-Base,   Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)** |   L=12,  H=768, A=12, d_1=3072   | 109482240 | 109.38M* bs * l_{max}   |
  | **[BERT-Large,   Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)** |   L=24, H=1024, A=16, d_1=4096   | 335141888 |   334.87M * bs * l_{max} |
  

You can get same result by this project or above formula.

  

# Requirements

- Python 3.6

- Pytorch 1.3.1

- Pandas 0.22.0

- NumPy 1.14.2

- Matplotlib 2.2.2

  

# References

* [torchstat](https://github.com/Swall0w/torchstat)
* [torchsummaryX](https://github.com/nmhkahn/torchsummaryX)

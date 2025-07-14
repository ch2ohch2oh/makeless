## Models

### Bigram

A bigram model predicts the next word based on the previous word. 
For each word, it will predict the next word based on the counts of 
`(current word, next word)` pairs in the training data.

This approach is __not scalable__ as for a n-gram model, the count
matrix has `{size of vocab}^n` number of elements. Moreoever, 
the matrix will be extremely sparse.

### MLP

MLP model use the embeddings of previous words in the context to predict the next word
using a feedforward neural network.

[mlp_loss](plots/mlp_loss_plot.png)
import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N,D = X.shape;

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W);
  y_hot_encoding = np.zeros(scores.shape);
  index = np.arange(N); # index to get the class score of each example
  y_hot_encoding[index,y] = 1;
  logC = np.max(scores,1)[:,None];
  prob = np.exp(scores + logC) / np.sum(np.exp(scores + logC),1)[:,None];
  log_prob = -y_hot_encoding * np.log(prob);
  loss = (np.sum(log_prob)) / N + reg * np.sum(W**2); 
  back_loss = prob-y_hot_encoding;
  dW = (X.transpose().dot(back_loss)/N + 2 * reg * W);

  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N,D = X.shape;
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W);
  y_hot_encoding = np.zeros(scores.shape);
  index = np.arange(N); # index to get the class score of each example
  y_hot_encoding[index,y] = 1;
  logC = np.max(scores,1)[:,None];
  prob = np.exp(scores + logC) / np.sum(np.exp(scores + logC),1)[:,None];
  log_prob = -y_hot_encoding * np.log(prob);
  loss = (np.sum(log_prob)) / N + reg * np.sum(W**2); 
  back_loss = prob-y_hot_encoding;
  dW = (X.transpose().dot(back_loss)/N + 2 * reg * W);

  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


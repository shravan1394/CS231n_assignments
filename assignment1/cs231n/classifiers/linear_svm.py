import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #print (dW)
        dW[:,j] = dW[:,j] + X[i];
        dW[:,y[i]] = dW[:,y[i]] - X[i]; # correct_class_score's influence on dW

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train;

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * (2 * W);
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W); # o/p = num train * Num classes
  y_hot_encoding = np.zeros(scores.shape);
  index = np.arange(X.shape[0]); # index to get the class score of each example
  y_hot_encoding[index,y] = 1;
  correct_class_scores = scores[index,y][:,None]; # casting (N,) to (N,1) for easier usage
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #indices = index * W.shape[1] + y;
  #mask = np.zeros(scores.shape, np.bool);
  #mask[index,y] = 1;
  #masked_scores = np.ma.masked_array(scores,mask=mask);
  # deleting the values of scores corresponding to correct classes and finding the margin
  #margin =  np.reshape(scores[~masked_scores.mask], (X.shape[0],W.shape[1]-1)) - correct_class_scores + 1;
  margin = scores - correct_class_scores + 1;
  margin[index,y] = 0; # setting all correct class score to zero
  loss = np.sum(margin[margin > 0]);
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  grad_part1 = X.transpose().dot((margin>0));# Adds the samples whose margins are +ve at each i,j of W 
  # part2 consists of samples multiplied by the no. of times the margin was +ve for that particular sample in the classes
  grad_part2 = np.sum(margin > 0,1)[:,None] * X; # no.of +ve margins found in each sample and mutiplied by the corresponfding sample
  grad_part2 = grad_part2.transpose().dot(y_hot_encoding); # samples of same class added together
  dW = grad_part1 - grad_part2; # grad found
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  # averaging the loss and gradient
  loss /= X.shape[0];
  dW /= X.shape[0];

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * (2 * W);
  

  return loss, dW

# Sentiment Analysis - Language Model - BERT

**Download the dataset from:**
https://www.kaggle.com/c/twitter-sentiment-analysis-ee505-au21/data

Change the paths of the datasets in the main function. It's recommended to use Google Colab.

**Pre-processing/Feature Engineering:**
Remove the symbols corresponding to urls like ‘https’, ‘//’. Replace the symbols corresponding to hashtags ‘#’ with ‘hashtags’, Replace the occurrences of ‘mentions’ like ‘@’ with ‘entities’, since a proper noun entity usually comes up. This is done by compiling regular expressions for pattern matching. Tokenize the text using pre-trained BERT embeddings. This includes breaking down individual words into multiple other words like ‘happiness’ to ’happy’. Addition of tokens for starting and ending the sentence also takes place using [CLS], [SEP] respectively. We set a max length for the tokens to be 64 for better computational efficiency during training. Additionally, attention masks are created to ensure that the model ignores the padded parts. This would improve the computation time during training for smaller tweets with less tokens. For our initial iterations, we do a 70:10:20 (train: val: test) split  on the dataset. However while evaluating on the unseen Kaggle data, we use the complete train dataset. Unfortunately, cross validation was not possible due to the excessive training time for a single train, validation combination.

**Model Building:**
Initial iterations were performed on 10% of the dataset and eventually the size was increased. Used Adam optimizer and performed hyperparameter tuning over the batch size and learning rate, until a final batch size of 32 and learning rate of 10. The number of epochs were increased as well from 1 to 2, however, this did not improve the performance of the model significantly. Since the entire architecture is partially pre-trained, the decoder block had to be modified for the purpose of sentiment analysis.

**Results:**
Evaluation on the test set:
Accuracy:  89.76%
Precision:  90.02%
Recall:  89.32%
f1 score:  89.7%

![](images/Loss%20Curve.png)

The data points in the loss curve have been computed every 2000 iterations, to reduce the computational load, since calculating the validation loss at every iteration was too time consuming.

**Insights:**
1. Tokenization takes semantics into account unlike traditional Bag of Words (BoW) based embeddings. The position of occurrence of words in a sentence is taken into account. 
2. Tuning a transformer based model requires almost no feature engineering and achieves state of the art performance with minimal hyperparameter tuning, this saves a lot of developer time. 
3. Based on the Figure, loss curve, after 4000 iterations, the validation loss was almost constant. Adding more data in subsequent batches did not improve performance the validation loss, infact, the train loss reduced, the model might have started to overfit. Early stopping would have helped in this regard. This also shows that we do not need a lot of data to achieve great predictions. 
4. On analyzing some of the incorrect samples, it’s difficult to know why the model predicted a certain sentence as negative sentiment or which word or word combination contributed to the prediction being incorrect. The interpretability of transformer based models is poor.  
5. If we want to deploy this solution in a system where the requirement is to have instantaneous predictions, it’s not recommended, as the process of tokenization and prediction of a new sentence doesn’t happen instantaneously.

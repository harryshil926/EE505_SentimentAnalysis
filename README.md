# EE505_SentimentAnalysis

Pre-processing/Feature Engineering: Remove the symbols corresponding to urls like 
‘https’, ‘//’. Replace the symbols corresponding to hashtags ‘#’ with ‘hashtags’, Replace the occurrences of ‘mentions’ like ‘@’ with ‘entities’, since a proper noun entity usually comes up. This is done by compiling regular expressions for pattern matching. Tokenize the text using pre-trained BERT embeddings. This includes breaking down individual words into multiple other words like ‘happiness’ to ’happy’. Addition of tokens for starting and ending the sentence also takes place using [CLS], [SEP] respectively. We set a max length for the tokens to be 64 for better computational efficiency during training. Additionally, attention masks are created to ensure that the model ignores the padded parts. This would improve the computation time during training for smaller tweets with less tokens. For our initial iterations, we do a 70:10:20 (train: val: test) split  on the dataset. However while evaluating on the unseen Kaggle data, we use the complete train dataset. Unfortunately, cross validation was not possible due to the excessive training time for a single train, validation combination.

Model Building:
Initial iterations were performed on 10% of the dataset and eventually the size was increased. Used Adam optimizer and performed hyperparameter tuning over the batch size and learning rate, until a final batch size of 32 and learning rate of 10. The number of epochs were increased as well from 1 to 2, however, this did not improve the performance of the model significantly. Since the entire architecture is partially pre-trained, the decoder block had to be modified for the purpose of sentiment analysis.

Results:
Evaluation on the test set:
Accuracy:  89.76%
Precision:  90.02%
Recall:  89.32%
f1 score:  89.7%

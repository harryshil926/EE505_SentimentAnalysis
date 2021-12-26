import pandas as pd
from google.colab import drive
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import re
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pylab import rcParams
! pip install transformers
from transformers import BertTokenizer,BertForSequenceClassification,AdamW
from tqdm.notebook import tqdm
from transformers import get_linear_schedule_with_warmup
from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import RandomSampler
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import numpy as np
import re
from pylab import rcParams
from transformers import BertTokenizer,BertForSequenceClassification,AdamW
from tqdm.notebook import tqdm
from transformers import get_linear_schedule_with_warmup
import random

class Data():
  def __init__(self):
      '''
      Define the train-validation split ratio.
      '''
      self.train_percent = 0.7
      self.val_percent = 0.1
      self.test_percent = 1 - self.train_percent - self.val_percent

  def load_data(self,path):
      '''
      Load the csv file.

      Parameters
      ----------
      path : Location where file is stored.
      '''
      self.df_data = pd.read_csv(path)

  def split_xy(self, choose_percentage_of_data=0.01):
      '''
      Split the dataset into X, y.

      Parameters
      ----------
      choose_percentage_of_data : amount of data to use for further modeling.
      '''
      self.X = self.df_data['tweet'].sample(frac=choose_percentage_of_data,random_state=seed)
      try:
        self.y = self.df_data['target'].sample(frac=choose_percentage_of_data,random_state=seed)
      except:
        self.y = pd.Series([0]*len(self.X))

  def pre_process_wrapper(self):
      '''
      Perform pre-processing on the text.
      '''
      def pre_process_text(text):
        hashtags = re.compile(r"^#\S+|\s#\S+")
        mentions = re.compile(r"^@\S+|\s@\S+")
        urls = re.compile(r"https?://\S+")
        text = re.sub(r'http\S+', '', text)
        text = hashtags.sub(' hashtag', text)
        text = mentions.sub(' entity', text)
        return text
      self.text = self.X.apply(pre_process_text).values
      self.target = self.y.values
      
  def tokenize(self):
      '''
      Tokenize using pre-trained BERT embeddings.
      '''
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
      self.input_ids = []
      self.attention_mask = []
      for i in self.text:
        encoded_data = tokenizer.encode_plus(
        i,
        add_special_tokens=True, # CLS and SEP tokens for BERT
        max_length=64,
        pad_to_max_length = True,
        return_attention_mask= True,
        return_tensors='pt')
        self.input_ids.append(encoded_data['input_ids'])
        self.attention_mask.append(encoded_data['attention_mask'])
      self.input_ids = torch.cat(self.input_ids,dim=0)
      self.attention_mask = torch.cat(self.attention_mask,dim=0)
      self.target = torch.tensor(self.target)
      self.dataset = TensorDataset(self.input_ids,self.attention_mask,self.target)
        
  
  def split_data(self, batch_size=32):
      '''
      Split the dataset intro train, val and test and create dataloaders
      for iterations.
    
      Parameters
      ----------
      batch_size : Total training samples to pass through the network, (Default=32)
      '''
      train_size = int(self.train_percent*len(self.dataset))
      val_size = int(self.val_percent*len(self.dataset))
      test_size = len(self.dataset)-train_size-val_size
      train_dataset, val_dataset, test_dataset = random_split(self.dataset,[train_size,val_size,test_size])
      self.train_dataset = self.create_dataloader(train_dataset, sampling = 'Random',batch_size=batch_size)
      self.val_dataset = self.create_dataloader(val_dataset, sampling = 'Sequential',batch_size=batch_size)
      self.test_dataset = self.create_dataloader(test_dataset, sampling = 'Sequential',batch_size=batch_size)

  def create_dataloader(self, dataset, sampling = 'Random',batch_size=32):
      '''
      Parent function for creating dataloaders.

      Parameters
      ----------
      sampling : Randomly shuffles the data samples - Train Set
                 Sequentially load the data samples - Validation/Test Set
      batch_size : Samples to load in each network propagation.

      '''
      if sampling == "Random":
        return DataLoader(dataset,sampler = RandomSampler(dataset),
                      batch_size = batch_size)
      else:
        return DataLoader(dataset,sampler = SequentialSampler(dataset),
                      batch_size = batch_size)

class Model(Data):
    def __init__(self,Data):
      '''
      Define the initialization parameters for the model, optimizer, epochs, schedulers.
      '''
      self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False)
      
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.model.to(self.device)
      self.DataObj = Data
      self.optimizer = AdamW(self.model.parameters())
      self.epochs = 1
      total_steps = len(self.DataObj.train_dataset)*self.epochs
      self.scheduler = get_linear_schedule_with_warmup(self.optimizer,num_warmup_steps=0,
                                           num_training_steps=total_steps)
      
    def get_evaluation_metrics(self, actuals, predictions):
      '''
      Evaluate the performance of the model.

      Parameters
      ----------
      actuals : The actual values of val/test set.
      predictions :  The predictions obtained on the val/test set.
      '''
      actuals = actuals.flatten()
      acc = np.sum(predictions==actuals)/len(actuals)
      f1 = f1_score(actuals, predictions)
      precision = precision_score(actuals, predictions)
      recall = recall_score(actuals, predictions)
      return {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1':f1}
      
    def train(self):
      '''
      Perform training using the previously created train set
      and compute the losses
      '''
      torch.cuda.empty_cache()
      self.loss_train_lst = []
      self.loss_val_lst = []
      for curr_epoch in tqdm(range(1, self.epochs+1)):
        self.model.train()
        loss_train_total = 0
        progress_bar = tqdm(self.DataObj.train_dataset, desc='Epoch {:1d}'.format(curr_epoch), leave=False, disable=False)
        for index, batch in enumerate(progress_bar):
          self.model.zero_grad()
          batch = tuple(b.to(self.device) for b in batch)
          inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                  }       
          outputs = self.model(**inputs)
          loss = outputs[0]
          loss_train_total += loss.item()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
          self.optimizer.step()
          self.scheduler.step()
          progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})   
          if index%1000==0:
            val_loss, predictions, actuals = self.predict(self.DataObj.val_dataset)
            self.loss_val_lst.append(val_loss)
            self.loss_train_lst.append(loss.item())

      tqdm.write(f'\nEpoch number: {curr_epoch}')
      loss_train_avg = loss_train_total/len(self.DataObj.train_dataset)            
      tqdm.write(f'Training loss: {loss_train_avg}')
      val_loss, predictions, actuals = self.predict(self.DataObj.val_dataset)
      metrics = self.get_evaluation_metrics(actuals, predictions)
      metrics, actuals, predictions = self.evaluate(dataset = 'val')
      tqdm.write(f'Validation loss: {val_loss}')
      tqdm.write(f'Validation Accuracy: {metrics["Accuracy"]}')
      tqdm.write(f'Validation F1-Score: {metrics["F1"]}')
      tqdm.write(f'Validation Precision: {metrics["Precision"]}')
      tqdm.write(f'Validation Recall {metrics["Recall"]}')

    def predict(self, dataloader_test):
      '''
      Get predictions on the val/test data.
      '''
      self.model.eval()
      loss_val_total = 0
      predictions,actuals = [],[]
      for batch in dataloader_test:
          batch = tuple(b.to(self.device) for b in batch)
          inputs = {
              'input_ids':batch[0],
              'attention_mask': batch[1],
              'labels': batch[2]
          }
          with torch.no_grad():
              outputs = self.model(**inputs)
          loss = outputs[0]
          logits = outputs[1]
          loss_val_total += loss.item()
          logits = logits.detach().cpu().numpy()
          label_ids = inputs['labels'].cpu().numpy()
          predictions.append(logits)
          actuals.append(label_ids)
      loss_val_avg = loss_val_total / len(dataloader_test)
      predictions = np.concatenate(predictions,axis=0)
      predictions = np.argmax(predictions,axis=1).flatten()
      actuals = np.concatenate(actuals,axis=0)
      return loss_val_avg,predictions,actuals

    def plot_loss_curve(self):
      '''
      Plot the train validation loss curve
      '''
      rcParams['figure.figsize'] = 8, 5
      plt.plot(self.loss_train_lst,label='Train')
      plt.xlabel('Iterations')
      plt.ylabel('Loss')
      plt.title('Train-Validation Loss Curve')
      plt.plot(self.loss_val_lst,label='Validation')
      plt.legend(loc="upper right")
      return plt

    def evaluate(self, dataset = 'val'):
      '''
      Dataset to evaluate performance on.

      Parameters
      ----------

      dataset: str, 'train', 'val', Default='val'
      '''
      hm = {}
      hm['val']=self.DataObj.val_dataset
      hm['test']=self.DataObj.test_dataset
      val_loss, predictions, actuals = self.predict(hm[dataset])
     
      metrics = self.get_evaluation_metrics(actuals, predictions)
      return metrics, actuals, predictions
    
    def predict_external(self, external_data):
      '''
      Get predictions on external unseen data
      '''

      val_loss, predictions, actuals = self.predict(external_data)
      return predictions

    def predict_external(self, external_data):
      '''
      Get predictions on external unseen data
      '''

      val_loss, predictions, actuals = self.predict(external_data)
      return predictions

if __name__ == "__main__":

  # For reproducibility
  seed_val = 7
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)
  drive.mount('/content/gdrive')
  # 1. Create an instance
  data = Data()
  # 2. Read data
  data.load_data('/content/gdrive/MyDrive/EE_505/Dataset/twitter_sentiments_data.csv')
  # 3. Split into X,y
  data.split_xy(choose_percentage_of_data=1)
  # 4. Perform Pre-processing
  data.pre_process_wrapper()
  # 5. Tokenize
  data.tokenize()
  # 6. Split the dataset
  data.split_data()
  # 7. Define the model parameters and train
  model = Model(data)
  model.train() 
  # 8. Evaluate on test set
  model.evaluate(dataset = 'test')
  # 9. Plot the loss curve
  model.plot_loss_curve()
  # 10. External Kaggle data predict
  external_data_obj = Data()
  external_data_obj.load_data('/content/gdrive/MyDrive/EE_505/Dataset/twitter_sentiments_evaluation.csv')
  external_data_obj.split_xy(choose_percentage_of_data=1)
  external_data_obj.pre_process_wrapper()
  external_data_obj.tokenize() 
  ext_loader = external_data_obj.create_dataloader(external_data_obj.dataset,sampling = 'Sequential',batch_size=32)
  predictions = model.predict_external(ext_loader)
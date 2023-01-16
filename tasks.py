import json
import pandas as pd
import numpy as np

# Natural language processing
import nltk
import re
from nltk.corpus import stopwords
from nltk.collocations import *
import string
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import unidecode
from word2number import w2n
import contractions
from nltk.tokenize import RegexpTokenizer

# Because we will use tensorflow
import tensorflow as tf

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Load Huggingface transformers
import transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFTrainer, Trainer
from transformers import AlbertConfig, AlbertTokenizerFast, TFAlbertModel, TrainingArguments

# load_data
ml_df = pd.read_csv('data/df_ml.csv')

# Select required columns
data = ml_df[['product','company_response_to_consumer',
            'consumer_disputed','clean_narrative', 'issue']]

# Remove a row if any of the remaining columns are missing
data = data.dropna(subset=['clean_narrative'])


# Renaming the data
df_sampled = data.copy()

# Remove rows, where the label is present only ones (can't be split)
df_sampled = df_sampled.groupby('product').filter(lambda x : len(x) > 1)
df_sampled = df_sampled.groupby('company_response_to_consumer').filter(lambda x : len(x) > 1)
df_sampled = df_sampled.groupby('consumer_disputed').filter(lambda x : len(x) > 1)
df_sampled = df_sampled.groupby('issue').filter(lambda x : len(x) > 1) 

# Set your model output as categorical and save in new label col
df_sampled['product_label'] = pd.Categorical(df_sampled['product'])
df_sampled['company_res_label'] = pd.Categorical(df_sampled['company_response_to_consumer'])
df_sampled['disputed_label'] = pd.Categorical(df_sampled['consumer_disputed'])
df_sampled['issue_label'] = pd.Categorical(df_sampled['issue'])

# Transform your output to numeric
df_sampled['product'] = df_sampled['product_label'].cat.codes
df_sampled['company_response_to_consumer'] = df_sampled['company_res_label'].cat.codes
df_sampled['consumer_disputed'] = df_sampled['disputed_label'].cat.codes
df_sampled['issue'] = df_sampled['issue_label'].cat.codes

# creating files
def get_dicts(df, col, label):
  items = df[col].unique().tolist()
  items.sort()
  item_cats = df[label].unique().categories.tolist()

  return dict(zip(items, item_cats))

# creating dictionaries to be dumped into json files later
product_dict = get_dicts(df = df_sampled, col ='product', label='product_label')
issue_dict = get_dicts(df=df_sampled, col='issue', label='issue_label')
disputed_dict = get_dicts(df=df_sampled, col='consumer_disputed', label='disputed_label')
company_res_dict = get_dicts(df=df_sampled, col='company_response_to_consumer', label = 'company_res_label')

# get json dicts
def get_labels(filename):
    with open(filename) as json_file:
        file = json.load(json_file)
    return file

# saving files to disk
def save_file(file, absolute_path, relative_path):
  """A function that saves files to disk"""
  full_path = absolute_path+relative_path
  with open(full_path, 'w') as convert_file:
    convert_file.write(json.dumps(file))

# Executing creation of files    
product = save_file(product_dict, 'files/', 'products.txt')
issue = save_file(issue_dict, 'files/', 'issues.txt')
disputed = save_file(disputed_dict, 'files/', 'disputed.txt')
response = save_file(company_res_dict, 'files/', 'company_response.txt')

def clean_text(text):
    """ A function to preprocess text"""
    #convert Non-ASCII characters to ASCII
    text = unidecode.unidecode(text)
    
    # Expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)

    # removes special characters from text, e.g. $"""
    clean = re.compile(r'[^a-zA-Z0-9\s]')
    text = re.sub(clean, ' ', text)
    # converts characters to lowercase"""
    text = text.lower()
    
    # remove patterns
    text = re.sub(r'xxxx', r'', text)
    
    # Convert number words to digits and remove them"""
    
    pattern = r'(\W+)'
    tokens = re.split(pattern, text)

    for i, token in enumerate(tokens):
        try:
            tokens[i] = str(w2n.word_to_num(token))
        except:
            pass
    
    text = ''.join(tokens)

    # removes numbers from text"""
    clean_1 = re.compile(r'\d+')
    
    text = re.sub(clean_1, '', text)

    # removes words with length 1 or 2"""
    clean = re.compile(r'\b\w{1,2}\b')
    
    text = re.sub(clean, '', text)
    
    # removes the stopwords in the text"""
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords = set(stopwords)
    
    tokens = re.split(r'(\W+)', text)
    
    text = [token for token in tokens if token not in stopwords]

    text = ' '.join(text)

    # Remove extra spaces from a string"""
    
    clean = re.compile(r'\s{2,10000}')
    text = re.sub(clean, ' ', text)


    lemma = WordNetLemmatizer()
    
    tokens = re.split('\W+', text)
    
    text = [lemma.lemmatize(token) for token in tokens]
    
    text = ' '.join(text)
    
    return text
    
class load_BERTModel(object):
   """ A class that loads the BertClass to be used in modelling"""
   def __init__(self, Bert_class, BERTModel_name, max_length, Bert_configuration, Bert_tokenizer):
      # Name of the BERT model to use
      self.BERTModel_name = BERTModel_name
      
      # Load transformers config and set output_hidden_states to False
      self.Bert_configuration = Bert_configuration.from_pretrained(BERTModel_name)
      self.Bert_configuration.output_hidden_states = False
      
      # Max length of tokens
      self.max_length = max_length
      
   def config(self, Bert_configuration):
      return self.Bert_configuration
      
   # Load BERT tokenizer
   def tokenize(self, Bert_tokenizer, BERTModel_name):
      Bert_tokenized = Bert_tokenizer.from_pretrained(pretrained_model_name_or_path = BERTModel_name, config = self.Bert_configuration)
      return Bert_tokenized
      
   # Load the Transformers BERT model
   def transform(self, Bert_class, BERTModel_name):
      transformer_model = Bert_class.from_pretrained(BERTModel_name, config = self.Bert_configuration)
      return transformer_model
  
class Build_BERTModel(load_BERTModel):
    """ A class that builds a BERT Model using TF Keras"""
    def __init__(self):
        super().__init__(self, transformer_model)
        pass
        # Loading the main layer
    def bert_model(transformer_model, max_length, config, df, x_col, y_1_label, y_2_label, y_3_label, y_4_label):
        bert = transformer_model.layers[0]

        # Build your model input
        input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
        # attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
        # inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        inputs = {'input_ids': input_ids}

        # Load the Transformers BERT model as a layer in a Keras model
        bert_model = bert(inputs)
        dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')

        pooled_output = dropout(bert_model, training=False)

        # Then build your model output
        product = Dense(units=len(df[y_1_label].value_counts()),
                        kernel_initializer=TruncatedNormal(stddev=config.initializer_range), 
                        name='product')(pooled_output)
        company_response = Dense(units=len(df[y_2_label].value_counts()),
                                 kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                                 name='company_response_to_consumer')(pooled_output)
        disputed = Dense(units=len(df[y_3_label].value_counts()),
                         kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                         name='consumer_disputed')(pooled_output)
        issue = Dense(units=len(df[y_4_label].value_counts()),
                      kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                      name='issue')(pooled_output)
        
        outputs = {'product': product, 'company_response': company_response,
                   'disputed':disputed, 'issue':issue}

        # And combine it all in a model object
        model = Model(inputs=inputs, outputs=outputs, name='ALBERT_MultiLabel_MultiClass')
        
        return model
   
class train_BERT(object):
    """ A class to train the model"""
    def __init__(self):
        pass

    def train_model(model, optim, max_length, tokenizer,
                   df_train, valid_split, batch_size, 
                   n_epochs, x_col, y_1, y_2, y_3, y_4):  
        """ Train the model """
        # Set an optimizer
        optimizer = optim(
            learning_rate=5e-05,
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0)

        # Set loss and metrics
        loss = {'product': CategoricalCrossentropy(from_logits = True),
                'company_response': CategoricalCrossentropy(from_logits = True),
                'disputed': CategoricalCrossentropy(from_logits=True),
                'issue': CategoricalCrossentropy(from_logits=True)}

        metric = {'product': CategoricalAccuracy('accuracy'),
                'company_response': CategoricalAccuracy('accuracy'),
                'disputed': CategoricalAccuracy('accuracy'),
                'issue': CategoricalAccuracy('accuracy')}

        # Compile the model
        model.compile(
            optimizer = optimizer,
            loss = loss, 
            metrics = metric)

        # Ready output data for the model
        y_product = to_categorical(df_train[y_1])
        y_company_response = to_categorical(df_train[y_2])
        y_disputed = to_categorical(df_train[y_3])
        y_issue = to_categorical(df_train[y_4])

        # Tokenize the input (takes some time)
        x = tokenizer(
            text=df_train[x_col].to_list(),
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True, 
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True)
        # Fit the model
        model_history = model.fit(
            # x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
            x={'input_ids': x['input_ids']},
            y={'product': y_product,
            'company_response': y_company_response,
            'disputed': y_disputed,
            'issue': y_issue},
            validation_split=valid_split,
            batch_size=batch_size,
            epochs=n_epochs)
    
class test_BERT(train_BERT):
    """ Evaluate the model on the test data"""
    def __init__(self):
        super().__init__()
    
    def test_model(max_length, model, df_test, tokenizer,
                  x_col, y_1, y_2, y_3, y_4):
                
        # Ready output data for the model
        test_y_product = to_categorical(df_test[y_1])
        test_y_company_response = to_categorical(df_test[y_2])
        test_y_disputed = to_categorical(df_test[y_3])
        test_y_issue = to_categorical(df_test[y_4])
        
        # Tokenize
        test_x = tokenizer(
        text=df_test[x_col].to_list(),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True, 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = False,
        verbose = True)
        
        # Run evaluation
        model_eval = model.evaluate(
        x={'input_ids': test_x['input_ids']},
        y={'issue': test_y_issue, 'product': test_y_product,
           'company_response': test_y_company_response,
           'consumer_disputed': test_y_disputed}
        )
        
# Load saved model
def load_model(filepath):
    """ A function that loads the saved model"""
    return tf.keras.models.load_model(filepath)

class TFBertMainLayer(tf.keras.layers.Layer):
    """ Loading the main layer"""
    def __init__(self, config, **kwargs):
        super(TFBertMainLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased', config=config)
        
    def call(self, inputs):
        return self.bert(inputs)
    
def prepare_data(input_text):
    """ A function to tokenize the text"""
    
        # # Load BERT tokenizer
    # Name of the BERT model to use
    model_name = 'bert-base-uncased'

    # Max length of tokens
    max_length = 100

    # Load transformers config and set output_hidden_states to False
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
    token = tokenizer.encode_plus(
      text = input_text,
      add_special_tokens=True,
      max_length=100,
      truncation=True,
      padding='max_length',
      return_tensors='tf',
      verbose = True)
    return {
     'input_ids':tf.cast(token.input_ids, tf.float64),
     'attention_mask': tf.cast(token.attention_mask, tf.float64)
  }
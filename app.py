from transformers import T5ForConditionalGeneration, T5TokenizerFast
from torch.utils.data import DataLoader
import streamlit as st
import pandas as pd
import torch
import os


# # Let us define the main page
st.markdown("Translation page 🔠")

# Dropdown for the translation type
translation_type = st.sidebar.selectbox("Translation Type", options=["French ➡️ Wolof", "Wolof ➡️ French"])

# define a dictionary of versions
models = {
    "Version ✌️": {
        "French ➡️ Wolof": {
            "checkpoints": "wolof_translate/checkpoints/t5_small_custom_train_results_fw_v4",
            "tokenizer": "wolof_translate/tokenizers/t5_tokenizers/tokenizer_v4.json",
            "max_len": None
        }
    },
    "Version ☝️": {
        "French ➡️ Wolof": {
            "checkpoints": "wolof_translate/checkpoints/t5_small_custom_train_results_fw_v3",
            "tokenizer": "wolof_translate/tokenizers/t5_tokenizers/tokenizer_v3.json",
            "max_len": 51
            },
        "Wolof ➡️ French": {
            "checkpoints": "wolof_translate/checkpoints/t5_small_custom_train_results_wf_v3",
            "tokenizer": "wolof_translate/trokenizers/t5_tokenizers/tokenizer_v3.json",
            "max_len": 51
        }
    }
}

# add special characters from Wolof
sp_wolof_chars = pd.read_csv('wolof_translate/data/wolof_writing/wolof_special_chars.csv')

# add definitions
sp_wolof_words = pd.read_csv('wolof_translate/data/wolof_writing/definitions.csv')

# let us add a callback functions to change the input text
def add_symbol_to_text():
    
    st.session_state.input_text += st.session_state.symbol

def add_word_to_text():
    
    word = st.session_state.word.split('/')[0].strip()
    
    st.session_state.input_text += word

# Dropdown for introducing wolof special characters
if translation_type == "Wolof ➡️ French":
    
    symbol = st.sidebar.selectbox("Wolof characters", key="symbol", options = sp_wolof_chars['wolof_special_chars'], on_change=add_symbol_to_text)
    
    word = st.sidebar.selectbox("Wolof words/Definitions", key="word", options = [sp_wolof_words.loc[i, 'wolof']+" / "+sp_wolof_words.loc[i, 'french'] for i in range(sp_wolof_words.shape[0])], on_change=add_word_to_text)

# Dropdown for the model version
version = st.sidebar.selectbox("Model version", options=["Version ☝️", "Version ✌️"])

# Recuperate the number of sentences to provide
temperature = st.sidebar.slider("How randomly need you the translated sentences to be from 0% to 100%", min_value = 0,
          max_value = 100)


# make the process
try:
    
    # recuperate the max length
    max_len = models[version][translation_type]['max_len']
    
    # let us get the best model
    @st.cache_resource
    def get_modelfw_v3():
        
        # recuperate checkpoints
        checkpoints = torch.load(os.path.join('wolof_translate/checkpoints/t5_small_custom_train_results_fw_v3', "best_checkpoints.pth"), map_location=torch.device('cpu'))
        
        # recuperate the tokenizer
        tokenizer_file = "wolof_translate/tokenizers/t5_tokenizers/tokenizer_v3.json"
        
        # initialize the tokenizer
        tokenizer = T5TokenizerFast(tokenizer_file=tokenizer_file)
        
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        
        # resize the token embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        model.load_state_dict(checkpoints['model_state_dict'])
        
        return model, tokenizer
    
    # @st.cache_resource
    def get_modelwf_v3():
        
        # recuperate checkpoints
        checkpoints = torch.load(os.path.join('wolof_translate/checkpoints/t5_small_custom_train_results_wf_v3', "best_checkpoints.pth"), map_location=torch.device('cpu'))
        
        # recuperate the tokenizer
        tokenizer_file = "wolof_translate/tokenizers/t5_tokenizers/tokenizer_v3.json"
        
        # initialize the tokenizer
        tokenizer = T5TokenizerFast(tokenizer_file=tokenizer_file)
        
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        
        # resize the token embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        model.load_state_dict(checkpoints['model_state_dict'])
        
        return model, tokenizer

    if version == "Version ☝️":
        
        if translation_type == "French ➡️ Wolof":
            
            model, tokenizer = get_modelfw_v3()
        
        elif translation_type == "Wolof ➡️ French":
            
            model, tokenizer = get_modelwf_v3() 

    # set the model to eval mode
    _ = model.eval()
    
    language = "Wolof" if translation_type == "French ➡️ Wolof" else "French"
    
    # Add a title
    st.header(f"Translate French sentences to {language} 👌")
    
    # Recuperate two columns
    left, right = st.columns(2)

    if translation_type == "French ➡️ Wolof":

        # recuperate sentences
        left.subheader('Give me some sentences in French: ')
    
    else:

        # recuperate sentences
        left.subheader('Give me some sentences in Wolof: ')

    # for i in range(number):
        
    left.text_input(f"- Sentence", key = f"input_text")

    # run model inference on all test data
    original_translations, predicted_translations, original_texts, scores = [], [], [], {}

    if translation_type == "French ➡️ Wolof":
        
        # print a sentence recuperated from the session
        right.subheader("Translation to Wolof:")
    
    else:
        
        # print a sentence recuperated from the session
        right.subheader("Translation to French:")

    # for i in range(number):
        
    sentence = st.session_state[f"input_text"] + tokenizer.eos_token
    
    if not sentence == tokenizer.eos_token:
    
        # Let us encode the sentences
        encoding = tokenizer([sentence], return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
        
        # Let us recuperate the input ids
        input_ids = encoding.input_ids
        
        # Let us recuperate the mask
        mask = encoding.attention_mask
        
        # Let us recuperate the pad token id
        pad_token_id = tokenizer.pad_token_id
        
        # perform prediction
        predictions = model.generate(input_ids, do_sample = False, top_k = 50, max_length = max_len, top_p = 0.90,
                                        temperature = temperature/100, num_return_sequences = 0, attention_mask = mask, pad_token_id = pad_token_id)
        
        # decode the predictions
        predicted_sentence = tokenizer.batch_decode(predictions, skip_special_tokens = True)
        
        # provide the prediction
        right.write(f"Translation: {predicted_sentence[0]}")
        
    else:
        
        # provide the prediction
        right.write(f"Translation: ")
    
except Exception as e:
    
    st.warning("The chosen model is not available yet !", icon = "⚠️")
    
    st.write(e)
    



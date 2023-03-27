import json 
import pickle
import nltk
import numpy as np 
import random
import tensorflow
from data_preprocessing import get_stem_words
ignore_words = ['?', '!',',','.', "'s", "'m"]

model = tensorflow.keras.models.load_model('./chatbot_model.h5')
intents = json.load(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))

def preprocess_user_input(userinput):
    input_word_token_1 = ntlk.word_tokenize(userinput)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words)
    input_word_token_2 = sorted(list(set(input_word_token_2)))
    bag = []
    bag_of_words = []

    for word in words :
        if word in input_word_token_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)

    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_prediction(userinput):
    inp = preprocessing_userinput(userinput)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])

    return prediction_class_label
def bot_response(userinput):
    predict_class_label = bot_class_prediction(userinput)
    predicted_class = classes[predicted_class_label]
    for intent in intents ['intents']:
        if intent['tag'] == predicted_class:
         bot_response = random.chocie(intent['ressponse'])

         return bot_response


print('Hi I am stella how can I help you')
while True :
    userinput = input('type your message here ')
    print('userinput', userinput)
    response = bot_response(userinput)
    print('botresponse', response)
from numpy import argmax
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from ModelTraining import *

BASE_DIR = 'C:/Users/Elie/Documents/CIS 511'

def generateDescription(model, wordTokenizer, imageFeatures, maxLength):
    inputSeq = 'START'
    for i in range(maxLength):
        tokenSeq = wordTokenizer.texts_to_sequences([inputSeq])[0]
        tokenSeq = pad_sequences([tokenSeq], maxlen=maxLength)
        Y_hat = model.predict([imageFeatures, tokenSeq], verbose=0)
        Y_hat = argmax(Y_hat)
        word = wordTokenizer.sequences_to_texts([[Y_hat]])[0]
        if word==None: break
        elif word == 'END' or word=='end': 
            inputSeq += ' ' + 'END'
            break
        else:
            inputSeq += ' ' + word
    return inputSeq


def evaluateModel(model, dfTest, featuresTest, tokenizer, maxLength):
    Y_correct, Y_pred = [], []
    for index in tqdm(dfTest.index):
        imageFeatures = featuresTest[dfTest.loc[index, 'image']]
        listOfCaptions = dfTest.loc[index, 'captions']
        Y_hat = generateDescription(model, tokenizer, imageFeatures, maxLength)
        Y_correct.append([caption.split() for caption in listOfCaptions])
        Y_pred.append(Y_hat.split())
    print(f'BLEU-1: {corpus_bleu(Y_correct, Y_pred, weights=(1,0,0,0))}')
    print(f'BLEU-2: {corpus_bleu(Y_correct, Y_pred, weights=(0.5,0.5,0,0))}')
    print(f'BLEU-3: {corpus_bleu(Y_correct, Y_pred, weights=(0.3,0.3,0.3,0))}')
    print(f'BLEU-4: {corpus_bleu(Y_correct, Y_pred, weights=(0.25,0.25,0.25,0.25))}')

if __name__ == '__main__':
    trainFileName = BASE_DIR + '/Flicker8k_Text/Flickr_8k.trainImages.txt'
    dfTrain = getData(trainFileName)
    features = getImageFeatures(dfTrain)
    tokenizer, vocabularySize, maxCaptionLength = getTokenizerAndVar(dfTrain)
    
    testFileName = BASE_DIR + '/Flicker8k_Text/Flickr_8k.testImages.txt'
    dfTest = getData(testFileName)
    featuresTest = getImageFeatures(dfTest)

    modelName = 'model9.h5'
    model = load_model(f'models/{modelName}')

    print(f'Evaluating Deep Learning Model {modelName}...')
    evaluateModel(model, dfTest, featuresTest, tokenizer, maxCaptionLength)
import pandas as pd
import numpy as np
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, Dropout, LSTM, Embedding
from keras.models import Model
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model, to_categorical

BASE_DIR = 'C:/Users/Elie/Documents/CIS 511' #Directory that contains the Flicker8k data

def addMarkers(listOfCaptions):
    result = []
    for caption in listOfCaptions:
        result.append('START ' + caption + ' END')
    return result

def getData(fileName):
    # Load training images
    dfTrain = pd.read_csv(fileName, header=None)
    dfTrain.columns = ['image']

    # Get descriptions of the images
    dfDesc = pd.read_csv('Descriptions.csv', header=0)

    # Group Descriptions by image and join with Train dataset
    dfDesc = dfDesc.groupby('image', as_index=False).agg({'image' : 'first', 'caption' : list})
    dfDesc = dfDesc.set_index('image', drop=True)
    dfTrain.loc[:,'captions'] = dfTrain['image'].transform(lambda x: dfDesc.loc[x,'caption'])

    # Add START and END markers to the captions
    dfTrain.loc[:,'captions'] = dfTrain['captions'].transform(addMarkers)
    return dfTrain

def getImageFeatures(df, fileName='Features.pkl'):
    # Get features dictionary and subset it according to the training data
    allFeatures = load(open(fileName, 'rb'))
    features = {img: allFeatures[img] for img in df['image']}
    return features

def getTokenizerAndVar(df):
    # Create Tokenizer for the captions
    descriptionList = []
    for captionList in df['captions']:
        descriptionList += captionList
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(descriptionList)
    
    # Helped variables for the model
    vocabularySize = len(tokenizer.word_index) + 1
    maxCaptionLength = max(len(caption.split(' ')) for caption in descriptionList)

    return tokenizer, vocabularySize, maxCaptionLength

def defineModel(maxCaptionLength, vocabularySize):
    # The below handle input from image features
    input1 = Input(shape=(4096,))
    dropout1 = Dropout(0.5)(input1)
    dense1 = Dense(256, activation='relu')(dropout1)

    # The below handle input from the word sequence (for the LSTM)
    input2 = Input(shape=(maxCaptionLength))
    embed1 = Embedding(vocabularySize, 256, mask_zero=True)(input2)
    dropout2 = Dropout(0.5)(embed1)
    lstm1 = LSTM(256)(dropout2)

    # The below join the two layers above to decode all the features (from photo + previous word)
    merger = add([dense1, lstm1])
    dense2 = Dense(256, activation='relu')(merger)
    output = Dense(vocabularySize, activation='softmax')(dense2)

    # Define the Deep Learning Model
    model = Model(inputs = [input1, input2], outputs = output)
    model.compile(loss='categorical_crossentropy', optimizer='nadam')

    # Uncomment the below for Logging purposes
    # print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model

def createTokenSequences(listOfDescription, wordTokenizer, image, maxLength, sizeOfVocabulary):
    X1, X2, y = [], [], []
    for caption in listOfDescription:
        tokenSeq = wordTokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(tokenSeq)):
            inputSeq = tokenSeq[:i]
            inputSeq = pad_sequences([inputSeq], maxlen = maxLength)[0]
            outputSeq = tokenSeq[i]
            outputSeq = to_categorical([outputSeq], num_classes=sizeOfVocabulary)[0]
            X1.append(image)
            X2.append(inputSeq)
            y.append(outputSeq)
    return np.array(X1), np.array(X2), np.array(y)

def modelDataGenerator(trainDf, wordTokenizer, featureDict, maxLength, sizeOfVocabulary):
    while True:
        for index in trainDf.index:
            imageFeatures = featureDict[trainDf.loc[index, 'image']][0]
            listOfCaptions = trainDf.loc[index, 'captions']
            inputImage, inputSeq, outputSeq = createTokenSequences(listOfCaptions, wordTokenizer, imageFeatures, maxLength, sizeOfVocabulary)
            # print(imageFeatures.shape, trainDf.loc[index, 'image'])
            yield [inputImage, inputSeq], outputSeq

if __name__ == '__main__':
    trainFileName = BASE_DIR + '/Flicker8k_Text/Flickr_8k.trainImages.txt'
    dfTrain = getData(trainFileName)
    features = getImageFeatures(dfTrain)
    tokenizer, vocabularySize, maxCaptionLength = getTokenizerAndVar(dfTrain)
    
    model = defineModel(maxCaptionLength, vocabularySize)
    epochs = 10
    steps = len(dfTrain['image'])
    for epoch in range(epochs):
        gen = modelDataGenerator(dfTrain, tokenizer, features, maxCaptionLength, vocabularySize)
        model.fit(gen, epochs = 1, steps_per_epoch = steps, verbose = 1)
        model.save('models/model' + str(epoch) + '.h5')

import pandas as pd
from tqdm import tqdm
from pickle import dump
from os.path import exists
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model


BASE_DIR = 'C:/Users/Elie/Documents/CIS 511' #Directory that contains the Flicker8k data
feature_dict = {}
vgg_model = VGG16()
vgg_model = Model(inputs = vgg_model.inputs, outputs = vgg_model.layers[-2].output)
tqdm.pandas(desc="Progress Bar")   

def preprocessImage(image_name, image_path):
    if image_name in feature_dict.keys():
        return feature_dict[image_name]
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    feature = vgg_model.predict(img)
    feature_dict[image_name] = feature
    return feature

def filterCaptions(caption):
    newCaption = []
    wordList = caption.split(' ')
    for word in wordList:
        word = word.lower()
        word = word.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''))
        if len(word)>1 and word.isalpha(): newCaption.append(word)
    return ' '.join(newCaption)

if __name__ == '__main__':
    df = pd.read_csv(f'{BASE_DIR}/captions.txt', header=0)

    # Extract the features from the images, save to feature_dict dictionary
    if not exists('Features.pkl'):
        print('Generating image features...')
        images = set(df['image'])
        for image in tqdm(images):
            preprocessImage(image, f'{BASE_DIR}/Images/{image}')
        dump(feature_dict, open('Features.pkl', 'wb'))

    # Extract vocabulary from captions
    if not exists('Descriptions.csv'):
        print('Cleaning up image descriptions...')
        df.loc[:,'caption'] = df['caption'].progress_apply(filterCaptions)
        df.to_csv('Descriptions.csv', index=False)
   
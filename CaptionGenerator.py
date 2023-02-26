from ModelTraining import *
from ModelTesting import *
from DataPreprocessing import *
from keras.models import load_model
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == '__main__':
    trainFileName = BASE_DIR + '/Flicker8k_Text/Flickr_8k.trainImages.txt'
    dfTrain = getData(trainFileName)
    features = getImageFeatures(dfTrain)
    tokenizer, vocabularySize, maxCaptionLength = getTokenizerAndVar(dfTrain)
    
    imageName = 'example.jpg'
    modelName = 'models/model9.h5'
    model = load_model(modelName)
    features = preprocessImage(imageName, str(Path().resolve())+'/'+imageName)
    caption = generateDescription(model, tokenizer, features, maxCaptionLength)
    print(caption)
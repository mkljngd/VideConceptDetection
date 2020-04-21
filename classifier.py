
#%%
import pickle
import cv2
import numpy as np
import os, subprocess
from sklearn.externals import joblib
# from memory_profiler import profile
import mahotas
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import sklearn.metrics as metrics
# from sklearn.model_selection import GridSearchCV

#%%
ucf101_path = "/Volumes/Data/MajorProject/JD/UCF-101"
trainlists = '/Volumes/Data/MajorProject/JD/UCF-101/ucfTrainTestlist'
trainlist01 = os.path.join(trainlists, 'trainlist01.txt')
testlist01 = os.path.join(trainlists, 'testlist01.txt')
training_output = '/Volumes/Data/MajorProject/JD/lol/tmp_frames/train'
testing_output = '/Volumes/Data/MajorProject/JD/lol/tmp_frames/test'

#%%
def extract_frame(videoName,frameName):
    """Doc
    Extracts the first frame from the input video (videoName)
    and saves it at the location (frameName)
    """
    #forces extracted frames to be 320x240 dim.
    if not os.path.exists(videoName):
        print (f'{videoName} does not exist!')
        return False
    # call ffmpeg and grab its stderr output
    p = subprocess.call(f'ffmpeg -i {videoName} -r 1 -s qvga -t 1 -f image2 {frameName}', shell=True)
    return p

def extract_frames(vidlist, vidDir, outputDir):
    f = open(vidlist, 'r')
    vids = f.readlines()
    f.close()
    vids = [video.rstrip() for video in vids]
    vids_dir = [line.split()[0].split('/')[0] for line in vids]
    vids = [line.split()[0].split('/')[1] for line in vids]
    for vid_dir, vid in zip(vids_dir, vids):
        videoName = os.path.join(vidDir, vid_dir, vid)
        frameName = os.path.join(outputDir, vid.split('.')[0]+".jpeg")
        extract_frame(videoName, frameName)
# extract_frames(trainlist01, ucf101_path,training_output)
# extract_frames(testlist01, ucf101_path,testing_output)

#%%
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
 
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0], None, [50], [0, 180])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

def make_frame_matrix(videos, vid_dir, class_index):
    global_features = []
    target = []
    for video in videos:
        vid_path = os.path.join(vid_dir, video)
        image = cv2.imread(vid_path)
        hist = fd_histogram(image)
        haralick = fd_haralick(image)
        moments = fd_hu_moments(image)
        global_feature = np.hstack([hist, haralick, moments])
        global_features.append(global_feature)
        name = video.split('_')[1]
        name = name.lower()
        target.append(class_index[name])
    scaler = MinMaxScaler(feature_range=(0, 1))
    #Normalize The feature vectors...
    rescaled_features = scaler.fit_transform(global_features)
    Y = np.array(target)
    return (rescaled_features, Y)

#%%
class_index_file = "class_index.npz"
class_index_file_loaded = np.load(class_index_file, allow_pickle=True)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]

training = [filename for filename in os.listdir(
    training_output) if filename.endswith('.jpeg') and (filename.__contains__('Archery') or filename.__contains__('BaseballPitch') or filename.__contains__('CricketBowling') or filename.__contains__('CricketShot') or filename.__contains__('Kayaking'))]
testing = [filename for filename in os.listdir(
    testing_output) if filename.endswith('.jpeg') and (filename.__contains__('Archery') or filename.__contains__('BaseballPitch') or filename.__contains__('CricketBowling') or filename.__contains__('CricketShot') or filename.__contains__('Kayaking'))]
test_txt = [x.split('.')[0] for x in testing]

#%%
X_train, Y_train = make_frame_matrix(training, training_output, class_index)
X_test, Y_test = make_frame_matrix(testing, testing_output, class_index)
np.savetxt('X_train.csv', X_train, delimiter=',')
np.savetxt('Y_train.csv', Y_train, delimiter=',')
np.savetxt('X_test.csv', X_test, delimiter=',')
np.savetxt('Y_test.csv', Y_test, delimiter=',')

#%%
classifier = SVC(C=0.66, kernel='rbf', gamma=0.55, shrinking=True, probability=True, tol=0.001,
                 cache_size=200, class_weight=None, verbose=True,
                 max_iter=-1, decision_function_shape='ovr', random_state=None).fit(X_train, Y_train)
joblib.dump(classifier, 'svm_model.pkl')
#s = pickle.dumps(classifier)
#%%
X_test_predictions = classifier.predict(X_test)
correct=np.zeros_like(X_test_predictions)
for k, (i, j) in enumerate(zip(X_test_predictions, Y_test)):
    if i==j:
        correct[k] = 1
np.savetxt('Predictions.csv', np.vstack((test_txt, Y_test, X_test_predictions, correct)).transpose(), '%s', ',', '\n', 'Video,Class,Prediction,Correct')
mean_accuracy = classifier.score(X_test, Y_test)
print(f'Correct: {np.count_nonzero(correct)}/{X_test.shape[0]}\nMean Accuracy: {mean_accuracy}')
# #%%
# svc = SVC(gamma='scale')
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(X_train, Y_train)

# #%%
# X_test_predictions=clf.predict(X_test)
# correct=0
# for i, j in zip(X_test_predictions, Y_test):
#     if i==j:
#         correct=correct + 1
# mean_accuracy = clf.score(X_test, Y_test)

#%%
train_txt = [x.split('.')[0] for x in training]
X_train_predictions = classifier.predict(X_train)
correct_train=np.zeros_like(X_train_predictions)
for k, (i, j) in enumerate(zip(X_train_predictions, Y_train)):
    if i==j:
        correct_train[k] = 1
np.savetxt('Train_Predictions.csv', np.vstack((train_txt, Y_train, X_train_predictions, correct_train)).transpose(), '%s', ',', '\n', 'Video,Class,Prediction,Correct')
train_mean_accuracy = classifier.score(X_train, Y_train)
print(f'Correct: {np.count_nonzero(correct_train)}/{X_train.shape[0]}\nMean Accuracy: {train_mean_accuracy}')

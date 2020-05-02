from sklearn.externals import joblib
import numpy as np
import math,os
import pandas as pd
from keras.models import load_model
import keras
import cv2
import subprocess
from damn import extract_frame, extract_frames, fd_hu_moments, fd_haralick, fd_histogram, make_frame_matrix


curr_path = os.getcwd()

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split() #ffmpeg to probe a video
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

def save_i_keyframes(video_fn,loc):
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I'] #select whether we want I frame,P frame or B frame
    cwd=os.getcwd()
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0].split('_')[1] #removing the extension
        cap = cv2.VideoCapture(video_fn)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            video_name = video_fn.split('.avi')[0]
            outname = video_name+'_i_frame.'+str(frame_no)+'.jpg'
            cv2.imwrite(outname, frame)
            print ('Saved: '+outname)
            break
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

classifier = joblib.load('svm_model.pkl')
class_index_file = "class_index.npz"
class_index_file_loaded = np.load(class_index_file, allow_pickle=True)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]


os.chdir('input')
for video in os.listdir('.'):
    save_i_keyframes(video,os.getcwd())

testing = [x for x in os.listdir('.') if x.find('.jpg') > 0]
testing_output = os.getcwd()
test_txt = [x.split('.')[0] for x in testing]
print(testing)
X_test, Y_test = make_frame_matrix(testing, testing_output, class_index)
np.savetxt('X_test.csv', X_test, delimiter=',')
np.savetxt('Y_test.csv', Y_test, delimiter=',')

X_test_predictions = classifier.predict(X_test)
correct=np.zeros_like(X_test_predictions)
for k, (i, j) in enumerate(zip(X_test_predictions, Y_test)):
    if i==j:
        correct[k] = 1
np.savetxt('Predictions.csv', np.vstack((test_txt, Y_test, X_test_predictions, correct)).transpose(), '%s', ',', '\n', 'Video,Class,Prediction,Correct')
mean_accuracy = classifier.score(X_test, Y_test)
print(f'Correct: {np.count_nonzero(correct)}/{X_test.shape[0]}\nMean Accuracy: {mean_accuracy}')

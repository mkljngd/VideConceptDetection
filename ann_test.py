import numpy as np
import math,os
import pandas as pd
from keras.models import load_model
import keras
from keyframe import LocalBinaryPatterns, get_frame_types
import cv2

curr_path = os.getcwd()
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



def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))

model = load_model('model.h5')

'''LOLOLOLOL'''
    
class_list=['Basketball','Bowling','GolfSwing','Kayaking','TennisSwing']


#os.chdir('./train')
histgram=[]
c=0
os.chdir('input')
print(os.listdir(os.getcwd()))
for video in os.listdir(os.getcwd()):
    if video.find('.avi')<=0: pass

    else:
        video_name = video.split('.avi')[0]
        save_i_keyframes(video,os.getcwd())
        lbpimg=LocalBinaryPatterns(8,1)
        for i in os.listdir(os.getcwd()): 
            gram=[]
            if(len(i.split("."))>1 and i.endswith('jpg')):
                print("THIS IS",i)
                #outn = './test_data/'+classes+str(c)+'.jpg'
                outn='lbp'+str(c)+'.jpg'
                print(outn)
                c=c+1
                imag=cv2.imread(i)
                gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
                img=lbpimg.imgLBP(gray,outn)
                c=c+1
                
                histo=lbpimg.describe(i)
                gram.append(histo)
                
                print(gram)
                arr=[i for i in np.nditer(gram)]
                #arr.append(d[classes])
                arr.append(0)
                histgram.append(arr)
np.savetxt('lbpdata_testlol.csv',histgram,delimiter=',')



test_ = pd.read_csv('lbpdata_testlol.csv',header=None)
test=test_.iloc[:,:10]
test.columns=['1','2','3','4','5','6','7','8','9','10']
X_test=test
classifier = load_model('model.h5')

y_class=[]
y_pred = classifier.predict(X_test)

for x in range(0,y_pred.shape[0]):
    row=[]
    clas=[]
    maxm=max(y_pred[x])
    for idx,i in enumerate(y_pred[x]):
        row.append(i)
        if(i==maxm):
            clas.append(idx+1)
    row=row+clas
    
    y_class.append(row)

np.savetxt('predictionsANN.csv',y_class,delimiter=",")

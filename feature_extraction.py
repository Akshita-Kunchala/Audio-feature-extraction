#1.Importing libraries and setup 
import os  
import librosa  
import librosa.display  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler  
import IPython.display as ipd  
from google.colab import auth 
from google.colab import drive 

auth.authenticate_user()
# Mount Google Drive
drive.mount('/content/drive',force_remount=True)
# Set file path
audio_path="/content/drive/MyDrive/audio_project/raw_audio/music/Dimple.mp3"  # Update path if needed
# Load audio file
y, sr=librosa.load(audio_path, sr=None)
# Optional: Play audio
ipd.display(ipd.Audio(audio_path))

#1.Plot waveform
plt.figure(figsize=(10,3))
librosa.display.waveshow(y,sr=sr)
plt.title("Waveform")
plt.tight_layout()
#Save the waveform visualization
plt.savefig("waveform.png")
plt.show()

#2.MFCC
mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
mfcc_scaled=StandardScaler().fit_transform(mfcc)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_scaled,x_axis='time',sr=sr)
plt.colorbar()
plt.title("MFCC (Scaled)")
plt.tight_layout()
#Save the mfcc visualization
plt.savefig("mfcc.png")
plt.show()

#3.Chroma
chroma=librosa.feature.chroma_stft(y=y,sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma,y_axis='chroma',x_axis='time',sr=sr)
plt.colorbar()
plt.title("Chroma Feature")
plt.tight_layout()
#Save the chroma visualization
plt.savefig("chroma.png")
plt.show()

#4.Spectral Contrast
contrast=librosa.feature.spectral_contrast(y=y,sr=sr)
plt.figure(figsize=(10,4))
librosa.display.specshow(contrast,x_axis='time',sr=sr)
plt.colorbar()
plt.title('Spectral Contrast')
plt.tight_layout()
#Save the spectral contrast visualization
plt.savefig("spectral_contrast.png")
plt.show()

#5.Extract single audio features
mfccs=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13).T,axis=0)
chroma=np.mean(librosa.feature.chroma_stft(y=y,sr=sr).T,axis=0)
contrast=np.mean(librosa.feature.spectral_contrast(y=y,sr=sr).T,axis=0)

#Combine all features into one vector
features=np.hstack([mfccs,chroma,contrast])
print("Feature vector shape:",features.shape)

#Save single audio features to CSV
feature_df=pd.DataFrame([features])
out_path="/content/drive/MyDrive/audio_project/outputs/audio_features.csv"
feature_df.to_csv(out_path,index=False)
print("Saved:",out_path)

"""
To download Manually
feature_df.to_csv("audio_features.csv",index=False)
from google.colab import files
files.download("audio_features.csv")
"""


#6.Define reusable function
def extract_features_from_file(file_path):
    y,sr=librosa.load(file_path,sr=None)
    mfccs=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13).T,axis=0)
    chroma=np.mean(librosa.feature.chroma_stft(y=y,sr=sr).T,axis=0)
    contrast=np.mean(librosa.feature.spectral_contrast(y=y,sr=sr).T,axis=0)
    return np.hstack([mfccs,chroma,contrast])

#7.Extract features for all WAV files in folder
folder="/content/drive/MyDrive/audio_project/raw_audio/music"
features=[]

for filename in os.listdir(folder):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        file_path=os.path.join(folder,filename)
        try:
            feat=extract_features_from_file(file_path)
            features.append(feat)
        except Exception as e:
            print(f"Error processing {filename}:{e}")

#Save all extracted features to CSV
df=pd.DataFrame(features)
df.to_csv(r"/content/drive/MyDrive/audio_project/outputs/all_audio_features.csv",index=False)
print("Saved all features Succesfully!")

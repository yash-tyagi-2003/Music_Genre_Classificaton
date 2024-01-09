import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('main_data.csv')
df=df.drop(['Unnamed: 0','instrumentalness','Unnamed: 9','Unnamed: 10'],axis=1)
df=df.dropna()
df=df[df.genre.isin(['hip hop'])==False]
for i in range(19890):
    df['danceability'][i]=float(df['danceability'][i])
    df['loudness'][i]=abs(float(df['loudness'][i]))
    df['acousticness'][i]=float(df['acousticness'][i])
    df['valence'][i]=float(df['valence'][i])
label_encoder = preprocessing.LabelEncoder()
genre=df['genre'].unique()
df['genre']= label_encoder.fit_transform(df['genre'])
X=df.drop(['genre','track_name'],axis=1)
y=df['genre']
model=KNeighborsClassifier(n_neighbors=50)
model.fit(X,y)
test_score = model.score(X,y)
with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
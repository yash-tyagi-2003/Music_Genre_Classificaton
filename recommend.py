import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings("ignore")
client_id = 'e42da97cd842452a8e2b8a62023e9a55'
client_secret = 'd881b55843c0432da08786e313735804'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
def get_val(name):
    results = sp.search(q=name, type='track', limit=1)
    track_uri = results['tracks']['items'][0]['uri']
    audio_features = sp.audio_features([track_uri])
    danceability = audio_features[0]['danceability']
    loudness = audio_features[0]['loudness']
    acousticness = audio_features[0]['acousticness']
    valence = audio_features[0]['valence']
    energy = audio_features[0]['energy']
    return [danceability,loudness,acousticness,valence,energy]
def classify(name):
    results = sp.search(q=name, type='track',limit=1)
    artist_name = results['tracks']['items'][0]['album']['artists'][0]['name']
    image = results['tracks']['items'][0]['album']['images'][0]['url']
    prev_link = results['tracks']['items'][0]['external_urls']['spotify']
    return [name.upper(),artist_name.upper(),image,prev_link]
def recommendation(name):
    fea=get_val(name)
    res=[]
    res.append(classify(name))
    df=pd.read_csv('main_data.csv')
    df=df.drop(['Unnamed: 0',"Unnamed: 9",'Unnamed: 10'],axis=1)
    df['all'] = df[['danceability', 'loudness', 'acousticness', 'valence', 'energy']].values.tolist()
    x = np.array(df['all'].tolist())
    cosine_similarities = cosine_similarity([fea], x)
    similar_indices = np.argsort(cosine_similarities[0])[::-1][:5]
    for i in range(5):
        res.append(classify(df['track_name'].iloc[similar_indices[i]].upper()))
    return res
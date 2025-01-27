import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pyarrow
lyrics_pd = pd.read_feather(
  'C:/Users/pauly/OneDrive/Documents/complete_lyrics_2025.feather'
)

gba = lyrics_pd.iloc[127]

gba_lyrics = gba['lyrics']

gba_lyrics

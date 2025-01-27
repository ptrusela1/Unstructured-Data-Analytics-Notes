import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pyarrow
lyrics_pd = pd.read_feather(
  'C:/Users/pauly/OneDrive/Documents/complete_lyrics_2025.feather'
)
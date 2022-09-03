from nrclex import NRCLex
import pandas as pd
import numpy as np

text = 'triste'
emotion = NRCLex(text)

print(emotion.words)
print(emotion.affect_dict)
print(emotion.raw_emotion_scores)
print(emotion.top_emotions)
print(emotion.affect_frequencies)

df = pd.read_csv('heehee.csv')
df['emotions'] = df['text'].apply(lambda x: NRCLex(x).affect_frequencies)
df = pd.concat([df.drop(['emotions'], axis = 1), df['emotions'].apply(pd.Series)], axis = 1)
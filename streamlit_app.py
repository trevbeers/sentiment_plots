import streamlit as st
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Sentiment:
    def __init__(self, variant):
        self.variant = variant
        self.labels = {
            'sadness': 'FOMO',
            'joy': 'excitement',
            'anger': 'action',
            'fear': 'intrigue'
        }

    def analyze(self):
        scores = {}

        tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        tokens = tokenizer([self.variant], return_tensors="pt")
        outputs = model(**tokens)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)

        predictions = {}
        for i, emotion in model.config.id2label.items():
            predictions[emotion] = prediction[0][i].item()

        for emotion, score in predictions.items():
            if emotion in self.labels:
                # Change emotion names
                scores[self.labels.get(emotion)] = score
            else:
                if emotion == 'love':  # add this score to the 'excitement' score
                    scores['excitement'] += score
                elif emotion == 'surprise':  # add this score to the 'intrigue' score
                    scores['intrigue'] += score

        return scores


# %%
st.title('Sentiment analysis')
# files = st.file_uploader('Add the files you want to analyze. Make sure the column you want to check is labeled '
#                          '"language"', type=['csv', 'xlsx'], accept_multiple_files=True, key='files')
sentence = st.text_input('Input text')
if sentence:
    emotion_scores = Sentiment(sentence).analyze()
    fig = px.bar(x=list(emotion_scores.values()), y=list(emotion_scores.keys()), orientation='h',
                 title=f'Text ⟶ "{sentence}"',
                 labels={'x': '', 'y': ''})
    fig.update_xaxes(range=[0, 1])
    st.write(fig)

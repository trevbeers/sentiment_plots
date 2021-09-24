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
            'fear': 'curiosity'
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
                    scores['curiosity'] += score

        return scores


# %%
st.title('Sentiment analysis')
# files = st.file_uploader('Add the files you want to analyze. Make sure the column you want to check is labeled '
#                          '"language"', type=['csv', 'xlsx'], accept_multiple_files=True, key='files')
sentence = st.text_input('Input text')
if sentence:
    emotion_scores = Sentiment(sentence).analyze()
    fig = px.bar(x=list(emotion_scores.values()), y=list(emotion_scores.keys()), orientation='h',
                 title=f'Input text ⟶ "{sentence}"',
                 labels={'x': '', 'y': ''})
    fig.update_xaxes(range=[0, 1])
    st.write(fig)

    st.header('How was this done?')
    st.write("""This classification was done using what is called a Transformer model. These models are trained on large 
    amounts of raw text — this initial training results in the models developing a probabilistic understanding of the 
    language the training data is written in. A model is then fine-tuned for specific tasks, emotion classification in 
    this case, using human-labeled data.""")

    st.header('What emotions did we look at?')
    st.write("""Curiosity, action, excitement and FOMO. Does the message intrigue people? Inspire them to act? Get them
    interested? Invoke FOMO? This model can tell you.""")

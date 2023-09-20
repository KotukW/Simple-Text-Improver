from flask import Flask, render_template, request
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])

def process():
    nltk.download('punkt')
    nlp = spacy.load("en_core_web_md")
    def error_correcting(text):
        tool = language_tool_python.LanguageTool('en-US')
        datasets = tool.correct(text)
        return datasets

    def split_sentence(sentence):
        doc = nlp(str(sentence))
        phrases = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return phrases

    def replace_phrase_in_sentence(sentence, phrase_to_replace, replacement):
        edited_sentence = sentence.replace(phrase_to_replace, replacement.lower())
        correct_text = error_correcting(edited_sentence)
        return correct_text

    def find_best_match(sentence, standard_phrases):
        best_standart_phrase_match = None
        best_sample_phrase_match = None
        best_similarity_score = 0.0
        
        for phrase in standard_phrases:
            for sample_data_phrase in split_sentence(sentence):
                similarity_score = cosine_similarity([nlp(sample_data_phrase).vector], [nlp(phrase).vector])[0][0]
                if similarity_score > best_similarity_score and similarity_score < 0.95:
                    best_similarity_score = similarity_score
                    best_standart_phrase_match = phrase
                    best_sample_phrase_match = sample_data_phrase
        return best_standart_phrase_match, best_similarity_score, best_sample_phrase_match

    user_text = request.form['text']
    user_input = list(request.form['replacement'])
    value_thresh = float(request.form['value_thresh'])

    sentences = sent_tokenize(user_text)
    results = []
    n = 0
    for sentence in sentences:
        best_match, similarity_score, best_data_phrase_match = find_best_match(sentence, user_input)
        if similarity_score >= value_thresh:
            if best_match:
                replacement = replace_phrase_in_sentence(sentence, best_data_phrase_match, best_match)
                results.append({
                    "original_sentence": sentence,
                    "improvement": replacement,
                    "similarity_score": float(similarity_score)
                })
            else:
                print(best_match)
                results.append({
                    "original_sentence": sentence,
                    "improvement": "No suitable match found.",
                    "similarity_score": 0.0
                })
        else:
            print(similarity_score)
            results.append({
                "original_sentence": sentence,
                "improvement": "No improvements needed.",
                "similarity_score": 0.0
            })
        n+=1
        print("OK!",n)

    return json.dumps(results)

if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
import pickle
import joblib

def dummy(text):
    return text
loaded_tokenizer = joblib.load(open('/Users/mahdiislam/Higher Studies/MAIA Study/Semester 1/Software Engineering/Project/Artificial_text_detection/tokenizer.pkl', 'rb'))
loaded_vectorizer = joblib.load(open('/Users/mahdiislam/Higher Studies/MAIA Study/Semester 1/Software Engineering/Project/Artificial_text_detection/tfidf_vectorizer.pkl', 'rb'))
loaded_model = pickle.load(open('/Users/mahdiislam/Higher Studies/MAIA Study/Semester 1/Software Engineering/Project/Artificial_text_detection/trained_model.sav', 'rb'))

def detection(input_data):

    test = []
    test.append(loaded_tokenizer.tokenize(input_data))
    X_test = loaded_vectorizer.transform(test)
    preds = loaded_model.predict(X_test)
    if preds == 0:
        return 'The text is not AI generated'
    else:
        return 'The text is AI generated'

def main():

    # giving a title 
    st.title('AI Text Detection App')

    # getting the data from user

    Input_cell = st.text_input('Write the text here')

    # code for prediction
    detect = ''

    if st.button('Detect'):
        detect = detection(Input_cell)

    st.success(detect)






if __name__ == '__main__':
    main()
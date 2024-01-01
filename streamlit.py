import streamlit as st
import pickle
import joblib
import re
import PyPDF2



alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def highlight_sentences(sentences, labels):
    highlighted_text = ""

    for sentence, label in zip(sentences, labels):
        if label >= 0.7:
            highlighted_text += f"<span style='color:red;'>{sentence}</span> "
        elif label >= 0.6 and label < 0.7:
            highlighted_text += f"<span style='color:grey;'>{sentence}</span> "
        elif label >= 0 and label < 0.6:
            highlighted_text += f"<span style='color:green;'>{sentence}</span> "

    return highlighted_text.strip()

def dummy(text):
    return text

loaded_tokenizer = joblib.load(open('/Users/mahdiislam/Higher Studies/MAIA Study/Semester 1/Software Engineering/Project/Artificial_text_detection/tokenizer.pkl', 'rb'))
loaded_vectorizer = joblib.load(open('/Users/mahdiislam/Higher Studies/MAIA Study/Semester 1/Software Engineering/Project/Artificial_text_detection/tfidf_vectorizer.pkl', 'rb'))
loaded_model = pickle.load(open('/Users/mahdiislam/Higher Studies/MAIA Study/Semester 1/Software Engineering/Project/Artificial_text_detection/trained_model.sav', 'rb'))

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText()
    return text

def detection(input_data):
    sentences = split_into_sentences(input_data)
    test = [loaded_tokenizer.tokenize(sentence) for sentence in sentences]
    X_test = loaded_vectorizer.transform(test)
    preds = [inner_list[1] for inner_list in loaded_model.predict_proba(X_test)]

    annotated_text = highlight_sentences(sentences, preds)

    ai_generated_count = sum(label >= 0.7 for label in preds)
    human_generated_count = sum(0 <= label < 0.6 for label in preds)
    uncertain_count = len(preds) - ai_generated_count - human_generated_count
    total_sentences = len(preds)

    ai_percentage = (ai_generated_count / total_sentences) * 100
    human_percentage = (human_generated_count / total_sentences) * 100
    uncertain_percentage = (uncertain_count / total_sentences) * 100


    mention = f"The text contains {ai_generated_count} AI-generated sentences ({ai_percentage:.2f}% - red), {human_generated_count} human-generated sentences ({human_percentage:.2f}% - green), and {uncertain_count} uncertain sentences ({uncertain_percentage:.2f}% - grey)."

    # mention = f"The text contains {ai_generated_count} AI-generated sentences (red), {human_generated_count} human-generated sentences (green), and {uncertain_count} uncertain sentences (grey)."

    return annotated_text, mention


def main():
    # giving a title
    st.title('AI Text Detection App')

    # Choose between text input and file upload
    option = st.radio("Choose Input Method", ("Text Input", "File Upload"))

    # Initialize input data
    input_data = ""

    # Handle text input
    if option == "Text Input":
        input_data = st.text_area('Write the text here', height=300)

    # Handle file upload
    elif option == "File Upload":
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                input_data = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                input_data = uploaded_file.read()

    
    # code for prediction
    annotated_text = ''
    mention = ''

    if st.button('Detect'):
        annotated_text, mention = detection(input_data)

        # Display mention
        st.markdown(mention, unsafe_allow_html=True)

        # Display highlighted text
        st.markdown("### Result")
        st.markdown(annotated_text, unsafe_allow_html=True)








if __name__ == '__main__':
    main()
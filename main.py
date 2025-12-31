# requirements: olama, streamlit, PyPDF2, langchain
# Assurez-vous que les modèles Olama sont installés localement (ex: llama-3)

import streamlit as st
from PyPDF2 import PdfReader
from olama import Olama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialisation du modèle Olama local
client = Olama(model="llama-3")  # Remplacez par le modèle installé

# Fonction pour extraire le texte d'un PDF
def extract_pdf_text(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

# Fonction pour générer un résumé
def generate_summary(text):
    prompt = PromptTemplate(
        input_variables=["doc"],
        template="Fais un résumé clair et structuré du document suivant:\n\n{doc}"
    )
    chain = LLMChain(prompt=prompt, llm=client)
    return chain.run(doc=text)

# Fonction pour répondre à des questions sur le document
def answer_question(text, question):
    prompt = PromptTemplate(
        input_variables=["doc", "question"],
        template="Document:\n{doc}\n\nQuestion: {question}\nRéponse détaillée:"
    )
    chain = LLMChain(prompt=prompt, llm=client)
    return chain.run(doc=text, question=question)

# Streamlit interface
st.title("Assistant de Recherche Académique avec Olama")
uploaded_file = st.file_uploader("Téléchargez un fichier PDF", type=["pdf"])

if uploaded_file:
    document_text = extract_pdf_text(uploaded_file)
    st.success("Document chargé avec succès!")

    if st.button("Générer un résumé"):
        summary = generate_summary(document_text)
        st.subheader("Résumé généré:")
        st.write(summary)

    question = st.text_input("Posez une question sur le document:")
    if question and st.button("Répondre à la question"):
        answer = answer_question(document_text, question)
        st.subheader("Réponse:")
        st.write(answer)

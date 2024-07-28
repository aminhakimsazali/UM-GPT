'''
requirements.txt file contents:
 
langchain==0.0.154 PyPDF2==3.0.1 python-dotenv==1.0.0 streamlit==1.18.1 faiss-cpu==1.7.4 streamlit-extras openai
'''
 
import re
# import PyPDF2
import numpy as np
from unidecode import unidecode
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# Sidebar contents
with st.sidebar:
    st.title('SC-GPT 🤖')
    st.markdown('''
    ## Weekend project by Amin Hakim (SC-GPT)
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/) as Frontend
    - [LLMStudios](https://lmstudio.ai/) as LLM-Backend
    ''')
    add_vertical_space(5)
    st.write('')
 
load_dotenv()


 
def main():


    def generateAnswerLLM(q):
        
        vs = []
        for i in tqdm(range(len(pdf_reader.pages))):
            c = cleaning(pdf_reader.pages[i].extract_text())
            v = get_embedding(c)
            vs.append(v)
        vs_np = np.array(vs)
        # q = 'What does Power of Commission to appoint statutory manager means? Explain in 50 words'
        q_v = get_embedding(q)
        score = cosine_similarity(vs_np, np.array([q_v]))[:,0]
        c_best_doc = cleaning(pdf_reader.pages[int(np.argmax(score))].extract_text())
        prompting = f'Text:  `{c_best_doc}`, \n\nBased on the text, act as Security Commissioner expert and answer the following question, `{q}`'

        completion = client.chat.completions.create(
        model="lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF",
        messages=[
            #     {"role": "system", "content": "Always answer in rhymes."},
                {"role": "user", "content": prompting}
            ],
            temperature=0.7,
        )
        # st.write(completion.choices[0].message.content)
        return completion.choices[0].message.content


    def cleaning(string):
        return re.sub(r'[ ]+', ' ', unidecode(string).replace('\n', ' ')).strip()

    def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    
    pdf_reader = PdfReader('Docs\Capital Markets and Services (Amendment) Act 2015.pdf')
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    v = get_embedding(cleaning(pdf_reader.pages[0].extract_text()))
    st.header("SC-GPT : Ask anything related to Securities Commission Malaysia regulations. ")  
    with st.expander("Disclaimer"):
        st.write("""Due to limited time and agile, this work limited to Regulatory FAQ provided by Malaysia SC.\n
        The documents that powered the LLM knowledge is accesible at 
        https://www.sc.com.my/regulation/regulatory-faqs
        """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = f"Echo: {prompt}"
            # response = generateAnswerLLM(prompt)
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    print(st.session_state.messages)

 
if __name__ == '__main__':
    main()
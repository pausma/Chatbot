import streamlit as st
import sys 
import os

os.environ["OPENAI_API_KEY"] = API_KEY
import openai
openai.api_key = API_KEY
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
)

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_all_internal_links(url):
    """Scrape a webpage and return all unique internal links."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract base URL for handling relative links
        base_url = "{0.scheme}://{0.netloc}".format(requests.utils.urlparse(url))

        internal_links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Join relative URLs with base URL
            if not href.startswith('http'):
                href = urljoin(base_url, href)
            # Add only internal links
            if href.startswith(base_url):
                internal_links.add(href)

        return list(internal_links)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

# URL to scrape
url = 'https://www.dolphinconsulting.cz/'

# Get all internal links from the website
internal_links = get_all_internal_links(url)


from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, ServiceContext, download_loader

service_context = ServiceContext.from_defaults(embed_model=embed_model)

BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=internal_links)


index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir='./cache/data/docs/')



chat_engine = index.as_chat_engine(chat_mode="condense_question")

st.title("Chatbot společnosti dolphin consulting a.s.")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Jak vám mohu pomoci?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


def generate_response(prompt_input):
    return chat_engine.chat(prompt_input).response      




if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Přemýšlím ... "):
            response = generate_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

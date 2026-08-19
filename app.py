"""Streamlit chat interface for the PCI DSS assistant.

Run it with:
    uv run streamlit run app.py
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv
from minsearch import VectorSearch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import ingest
from rag_helper import REQ_NUMBER_RE, RAGHybrid

EMBEDDING_MODEL = 'multi-qa-MiniLM-L6-cos-v1'

load_dotenv()

st.set_page_config(page_title='PCI DSS Assistant', page_icon='🔒', layout='centered')


@st.cache_resource(show_spinner='Loading the standard and building the indexes...')
def load_assistant():
    """Built once per server process, not once per interaction.

    Streamlit re-runs the whole script on every user action. Without this cache the
    261 pages would be re-embedded on every question, which takes about a minute.
    """
    ingest.download_pdf()
    documents = ingest.load_documents()

    text_index = ingest.build_index(documents)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vectors = embedder.encode([d['text'] for d in documents], batch_size=32)

    vector_index = VectorSearch()
    vector_index.fit(vectors, documents)

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), max_retries=8)

    assistant = RAGHybrid(
        text_index=text_index,
        vector_index=vector_index,
        embedder=embedder,
        llm_client=client,
    )

    return assistant, documents


def answer_question(assistant, question):
    started = time.time()

    search_results = assistant.search(question)
    prompt = assistant.build_prompt(question, search_results)
    answer = assistant.llm(prompt)

    return {
        'question': question,
        'answer': answer,
        'sources': search_results,
        # which branch of the router handled it — useful when an answer looks wrong
        'route': 'text (requirement number)' if REQ_NUMBER_RE.search(question) else 'hybrid',
        'elapsed': time.time() - started,
    }


def render_sources(result):
    label = f"Sources — {len(result['sources'])} pages, retrieved via {result['route']}"

    with st.expander(label):
        st.caption(f"answered in {result['elapsed']:.1f}s")

        for doc in result['sources']:
            st.markdown(
                f"**Requirement {doc['req_ids']}** · page {doc['printed_page']} "
                f"of the standard (PDF page {doc['page']})"
            )
            st.text(doc['text'][:700] + '...')


assistant, documents = load_assistant()

st.title('🔒 PCI DSS Assistant')
st.caption(
    f'Ask about the PCI DSS v4.0.1 standard. Answers come only from the '
    f'{len(documents)} pages of its "Requirements and Testing Procedures" section, '
    f'with the requirement number and page cited so you can check them.'
)

with st.sidebar:
    st.subheader('Try one of these')

    examples = [
        'How long must audit logs be retained?',
        'Do we need MFA for administrative access, or only for remote access?',
        'Can we store the CVV after the transaction is authorized?',
        'What exactly does requirement 8.3.6 ask for?',
    ]

    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

    st.divider()
    st.caption(f'Embeddings: `{EMBEDDING_MODEL}` (local)')
    st.caption(f'LLM: `{assistant.model}`')

if 'history' not in st.session_state:
    st.session_state.history = []

for result in st.session_state.history:
    with st.chat_message('user'):
        st.write(result['question'])

    with st.chat_message('assistant'):
        st.write(result['answer'])
        render_sources(result)

question = st.chat_input('Ask about PCI DSS...') or st.session_state.pop('pending_question', None)

if question:
    with st.chat_message('user'):
        st.write(question)

    with st.chat_message('assistant'):
        with st.spinner('Searching the standard...'):
            result = answer_question(assistant, question)

        st.write(result['answer'])
        render_sources(result)

    st.session_state.history.append(result)

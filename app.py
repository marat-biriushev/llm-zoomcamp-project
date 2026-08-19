"""Streamlit chat interface for the PCI DSS assistant.

Run it with:
    uv run streamlit run app.py
"""

import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from minsearch import VectorSearch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import db
import ingest
from rag_helper import REQ_NUMBER_RE, RAGHybridWithUsage, calc_price

EMBEDDING_MODEL = 'multi-qa-MiniLM-L6-cos-v1'

load_dotenv()

st.set_page_config(page_title='PCI DSS Assistant', page_icon='🔒', layout='centered')


@st.cache_resource(show_spinner='Loading the standard and building the indexes...')
def load_assistant():
    """Built once per server process, not once per interaction.

    Streamlit re-runs the whole script on every user action. Without this cache the
    261 pages would be re-embedded on every question, which takes about a minute.
    """
    # Prefer the pages the dlt pipeline loaded into Postgres. Falling back to
    # parsing the PDF keeps the app runnable on its own, without the pipeline.
    documents = db.load_documents()
    source = 'Postgres (dlt pipeline)'

    if documents is None:
        ingest.download_pdf()
        documents = ingest.load_documents()
        source = 'the PDF, parsed at startup'

    text_index = ingest.build_index(documents)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vectors = embedder.encode([d['text'] for d in documents], batch_size=32)

    vector_index = VectorSearch()
    vector_index.fit(vectors, documents)

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), max_retries=8)

    assistant = RAGHybridWithUsage(
        text_index=text_index,
        vector_index=vector_index,
        embedder=embedder,
        llm_client=client,
    )

    return assistant, documents, source


@st.cache_resource
def database_ready():
    """Create the tables once, and remember whether Postgres is reachable.

    The assistant stays usable without a database — monitoring is a nice-to-have,
    not a reason to refuse to answer questions.
    """
    try:
        db.init_db()
        return True
    except Exception as exc:  # noqa: BLE001 - any connection problem means "no logging"
        st.session_state.db_error = str(exc)
        return False


def answer_question(assistant, question):
    started = time.time()

    search_results = assistant.search(question)
    prompt = assistant.build_prompt(question, search_results)
    answer = assistant.llm(prompt)

    return {
        'id': str(uuid.uuid4()),
        'question': question,
        'answer': answer,
        'sources': search_results,
        # which branch of the router handled it — useful when an answer looks wrong
        'route': 'text (requirement number)' if REQ_NUMBER_RE.search(question) else 'hybrid',
        'elapsed': time.time() - started,
        'usage': assistant.last_usage,
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


def render_feedback(result):
    """Thumbs up / down. Keyed by conversation id so every answer keeps its own state."""
    conversation_id = result['id']
    given = st.session_state.votes.get(conversation_id)

    if given is not None:
        st.caption('Thanks — feedback recorded.' if given > 0 else 'Thanks — noted.')
        return

    left, right, _ = st.columns([1, 1, 8])

    for column, vote, label in ((left, 1, '👍'), (right, -1, '👎')):
        if column.button(label, key=f'{vote}-{conversation_id}'):
            st.session_state.votes[conversation_id] = vote

            if logging_enabled:
                db.save_feedback(conversation_id, vote)

            st.rerun()


assistant, documents, document_source = load_assistant()
logging_enabled = database_ready()

if 'history' not in st.session_state:
    st.session_state.history = []
if 'votes' not in st.session_state:
    st.session_state.votes = {}

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
    st.caption(f'Documents: {len(documents)} pages from {document_source}')

    if logging_enabled:
        st.caption('Monitoring: [Grafana](http://localhost:3000) · logging to Postgres')
    else:
        st.caption('Monitoring: off — Postgres unreachable, answers are not logged')

for past in st.session_state.history:
    with st.chat_message('user'):
        st.write(past['question'])

    with st.chat_message('assistant'):
        st.write(past['answer'])
        render_sources(past)
        render_feedback(past)

question = st.chat_input('Ask about PCI DSS...') or st.session_state.pop('pending_question', None)

if question:
    with st.chat_message('user'):
        st.write(question)

    with st.chat_message('assistant'):
        with st.spinner('Searching the standard...'):
            result = answer_question(assistant, question)

        st.write(result['answer'])
        render_sources(result)

        if logging_enabled:
            cost = calc_price(result['usage'])['total_cost']
            db.save_conversation(result['id'], result, assistant.model, result['usage'], cost)

        render_feedback(result)

    st.session_state.history.append(result)

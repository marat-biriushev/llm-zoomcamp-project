import re

# Prices per million tokens for the model used below. Update if you switch models.
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50

# Chosen in step 7 over two alternatives, on 100 questions judged by an LLM against
# the source page:
#
#                good   citation correct   no citation
#   minimal      0.78         0.76            0.21
#   structured   0.74         0.97            0.01   <- this one
#   strict       0.66         0.96            0.02
#
# `minimal` looks best on answer quality, but a paired comparison puts it level with
# `structured` (11 wins to 7, p = 0.48) — that gap was noise. It leaves one answer in
# five with no citation at all, which for a compliance question means an answer nobody
# can verify. `strict` asked for a citation after every individual claim and scored
# genuinely worse on substance (loses to `minimal` 1:13, p = 0.002): demanding
# inline citations everywhere fragments the answer.
INSTRUCTIONS = '''
You are an assistant that answers questions about the PCI DSS v4.0.1 standard.

Answer using ONLY the context below. If the answer is not in the context, reply
exactly "I don't know." Never invent a requirement number.

Structure every answer as:
1. One sentence stating the requirement in plain language.
2. The specifics — thresholds, timeframes, exceptions — as short bullet points.
3. A final line "Source: req. X.Y.Z (p. N)" listing the requirements you used.
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        # No boost_dict on purpose. Step 5 measured weights from 0.5 to 10 on
        # `req_ids` and every one of them scored identically: a requirement number
        # matches exactly one page, so any positive weight already puts it first.
        # Removing the field altogether is what hurts (hit rate 0.97 -> 0.84 on
        # questions that quote a number), so the field stays and the weight goes.
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            # cite the page number printed in the standard, not the PDF page index
            page = doc.get('printed_page', doc['page'])
            lines.append(f"[requirement {doc['req_ids']}, page {page}]")
            lines.append(doc['text'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer


class RAGVector(RAGBase):
    """Same pipeline, but retrieval goes through embeddings instead of TF-IDF."""

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)

        return self.index.search(query_vector, num_results=num_results)


def reciprocal_rank_fusion(result_lists, k=60, num_results=5):
    """Merge several ranked lists of documents into one.

    Scores from TF-IDF and from embedding cosine similarity live on different
    scales, so adding them together is meaningless. RRF sidesteps that by
    combining *positions* instead of scores:

        score(doc) = sum over lists of 1 / (k + rank)

    A document ranked first in both lists scores 1/61 + 1/61; one found by a
    single retriever scores 1/61. k=60 is the conventional value — it flattens
    the difference between the top few ranks so that a document has to do well
    in both lists to win.
    """
    scores = {}
    documents = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc['page']
            documents[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank + 1)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)

    return [documents[doc_id] for doc_id in ranked_ids[:num_results]]


# Requirement numbers run from 1.1 to 12.10.7, plus A1.x / A2.x / A3.x, and go at
# most five levels deep. Every component in the actual standard is between 1 and 12,
# which is what keeps "PCI DSS 4.0.1" (a zero) and "3.50 dollars" (a 50) from being
# mistaken for requirement numbers and routed to the wrong retriever.
_PART = r'(?:1[0-2]|[1-9])'
REQ_NUMBER_RE = re.compile(
    # appendix ids ("A1.1", "A3.2.1") need only one part after the prefix,
    # ordinary ones ("1.1", "10.2.1.1") need at least two parts overall
    rf'\b(?:A[1-3]\.{_PART}(?:\.{_PART}){{0,3}}|{_PART}(?:\.{_PART}){{1,4}})\b'
)


class RAGHybrid(RAGBase):
    """Text search and vector search, merged with reciprocal rank fusion.

    With `route_numbers=True` a question that quotes a requirement number skips
    fusion and goes to text search alone. Step 6 measured why: embeddings are
    useless on "8.3.6" (hit rate 0.68 against 0.87 on plain questions), so mixing
    their ranking in pushes the one exactly matching page *down*. Text search
    alone scores MRR 0.89 on those questions; the hybrid manages 0.75.
    """

    def __init__(self, text_index, vector_index, embedder, candidates=5,
                 route_numbers=True, **kwargs):
        super().__init__(index=text_index, **kwargs)
        self.vector_index = vector_index
        self.embedder = embedder
        # How many candidates each retriever contributes to the fusion. Measured at
        # 5, 10, 20 and 30: MRR moved by 0.006 with no clear direction, so this is
        # noise, not a knob. 5 is the cheapest of the tied options.
        self.candidates = candidates
        self.route_numbers = route_numbers

    def search(self, query, num_results=5):
        if self.route_numbers and REQ_NUMBER_RE.search(query):
            return self.index.search(query, num_results=num_results)

        text_results = self.index.search(query, num_results=self.candidates)

        query_vector = self.embedder.encode(query)
        vector_results = self.vector_index.search(query_vector, num_results=self.candidates)

        return reciprocal_rank_fusion(
            [text_results, vector_results],
            num_results=num_results
        )


def calc_price(usage):
    input_cost = (usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION

    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': input_cost + output_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        total_cost = total_cost + calc_price(usage)['total_cost']

    return total_cost


class UsageTracking:
    """Mixin that records token usage for every LLM call.

    Deliberately inherits from nothing. A mixin that also inherited RAGBase would
    break under Jupyter's `%autoreload`: reloading recreates the classes, so the
    RAGBase behind this mixin and the RAGBase behind RAGHybrid end up being two
    different objects, and the method resolution order silently puts the wrong one
    first. Keeping the mixin base-free makes every combination a simple chain.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)


class RAGWithUsage(UsageTracking, RAGBase):
    """Plain text-search RAG that also keeps track of what it spent."""


class RAGHybridWithUsage(UsageTracking, RAGHybrid):
    """The routed hybrid retrieval from step 6, with token accounting."""

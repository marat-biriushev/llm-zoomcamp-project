import time

from tqdm.auto import tqdm

from rag_helper import RAGBase

# Prices per million tokens for the model we use. Update if you switch models.
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50


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


def llm_structured(client, instructions, user_prompt, output_type, model='gpt-5.4-mini'):
    """Ask the model for an answer shaped like `output_type`, a pydantic model."""
    messages = [
        {'role': 'developer', 'content': instructions},
        {'role': 'user', 'content': user_prompt}
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type
    )

    return response.output_parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model='gpt-5.4-mini',
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(client, instructions, user_prompt, output_type, model=model)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):
    """Same as RAGBase, but remembers how many tokens every call cost."""

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


def hit_rate(relevance_total):
    """Share of questions where the right document made it into the results at all."""
    cnt = 0

    for line in relevance_total:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


def mrr(relevance_total):
    """Mean Reciprocal Rank — rewards the right document being near the top.

    Rank 1 scores 1.0, rank 2 scores 0.5, rank 3 scores 0.33, not found scores 0.
    """
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance_total)


def compute_relevance(q, search_function):
    doc_id = q['document']
    results = search_function(query=q['question'])

    relevance = []

    for d in results:
        relevance.append(int(d['page'] == doc_id))

    return relevance


def compute_relevance_total(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        relevance_total.append(compute_relevance(q, search_function))

    return relevance_total


def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


def map_progress(pool, seq, f):
    """Run f over seq in parallel, with a progress bar, keeping the input order."""
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            results.append(future.result())

    return results

"""Helpers used by the evaluation notebook.

The RAG classes and the pricing helpers live in `rag_helper` because the Streamlit
app needs them too; they are re-exported here so the notebook keeps working with a
single import.
"""

import time

from tqdm.auto import tqdm

from rag_helper import (  # noqa: F401 - re-exported for the notebook
    RAGHybridWithUsage,
    RAGWithUsage,
    calc_price,
    calc_total_price,
)


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

INSTRUCTIONS = '''
You are an assistant that answers questions about the PCI DSS v4.0.1 standard.

Answer using ONLY the context below. The context contains pages of the standard.
If the answer is not in the context, say "I don't know." Do not rely on general
knowledge about payment security, and never invent a requirement number.

Cite the requirement number and page for every claim you make,
like this: (req. 8.3.6, p. 194).
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
            lines.append(f"[requirement {doc['req_ids']}, page {doc['page']}]")
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

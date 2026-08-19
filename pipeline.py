"""Ingestion pipeline: PDF -> Postgres, orchestrated by dlt.

Why a pipeline at all
---------------------
Until now the Streamlit app downloaded and parsed the standard itself on every
cold start. That works, but it welds two unrelated jobs together: getting the data
in, and serving questions. Splitting them means the app boots in seconds against a
table, and re-ingesting a new revision of the standard never touches the app.

What dlt adds over a plain script: schema inference and creation, typed columns,
a `merge` write disposition keyed on the page number (so re-running updates rows
instead of duplicating them), and a load history you can inspect afterwards with
`dlt pipeline pci_dss info`.

Usage:
    uv run python pipeline.py
"""

import os

import dlt
from dlt.destinations import postgres
from dotenv import load_dotenv

import ingest

load_dotenv()

DATASET = 'pci_dss'
TABLE = 'pages'


def connection_string():
    user = os.getenv('POSTGRES_USER', 'pci')
    password = os.getenv('POSTGRES_PASSWORD', 'pci')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'pci')

    return f'postgresql://{user}:{password}@{host}:{port}/{database}'


@dlt.resource(
    name=TABLE,
    primary_key='page',
    # merge, not append: the standard is a fixed document, so a second run should
    # leave the table with 261 rows, not 522
    write_disposition='merge',
)
def pci_pages():
    """Yield one record per page of the Requirements and Testing Procedures section."""
    path = ingest.download_pdf()
    yield from ingest.load_documents(path)


def run():
    pipeline = dlt.pipeline(
        pipeline_name='pci_dss',
        destination=postgres(credentials=connection_string()),
        dataset_name=DATASET,
    )

    info = pipeline.run(pci_pages())
    print(info)

    return info


if __name__ == '__main__':
    run()

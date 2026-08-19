"""Postgres storage for conversations, retrieved sources and user feedback.

Three tables rather than one, because they answer different questions:

- `conversations` — one row per answer, with cost and latency. Drives most of the
  dashboard.
- `retrievals`   — one row per retrieved page. Lets the dashboard show *which parts
  of the standard people actually ask about*, which a single comma-separated column
  could not.
- `feedback`     — thumbs up or down, written later than the answer it refers to,
  hence a separate table rather than a nullable column.
"""

import os
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

SCHEMA = '''
CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    asked_at        TIMESTAMPTZ NOT NULL,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    route           TEXT NOT NULL,
    model           TEXT NOT NULL,
    response_time   DOUBLE PRECISION NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cost            DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS retrievals (
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    rank            INTEGER NOT NULL,
    page            INTEGER NOT NULL,
    printed_page    INTEGER NOT NULL,
    req_ids         TEXT NOT NULL,
    PRIMARY KEY (conversation_id, rank)
);

CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    vote            SMALLINT NOT NULL,
    given_at        TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS conversations_asked_at_idx ON conversations (asked_at);
CREATE INDEX IF NOT EXISTS feedback_given_at_idx ON feedback (given_at);
'''


def connection_string():
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'pci')} "
        f"user={os.getenv('POSTGRES_USER', 'pci')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'pci')}"
    )


def connect():
    return psycopg.connect(connection_string(), row_factory=dict_row)


def init_db():
    with connect() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def save_conversation(conversation_id, result, model, usage, cost):
    now = datetime.now(timezone.utc)

    with connect() as conn:
        conn.execute(
            '''
            INSERT INTO conversations
                (id, asked_at, question, answer, route, model,
                 response_time, input_tokens, output_tokens, cost)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            ''',
            (
                conversation_id,
                now,
                result['question'],
                result['answer'],
                result['route'],
                model,
                result['elapsed'],
                usage.input_tokens,
                usage.output_tokens,
                cost,
            ),
        )

        for rank, doc in enumerate(result['sources'], start=1):
            conn.execute(
                '''
                INSERT INTO retrievals (conversation_id, rank, page, printed_page, req_ids)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (conversation_id, rank) DO NOTHING
                ''',
                (conversation_id, rank, doc['page'], doc['printed_page'], doc['req_ids']),
            )

        conn.commit()


def load_documents():
    """Read the parsed pages that the dlt pipeline loaded.

    Returns None when the table is absent or empty, which lets the app fall back to
    parsing the PDF itself instead of refusing to start.
    """
    try:
        with connect() as conn:
            rows = conn.execute(
                '''
                SELECT page, printed_page, req_ids, requirement, text
                FROM pci_dss.pages
                ORDER BY page
                '''
            ).fetchall()
    except psycopg.Error:
        return None

    return [dict(row) for row in rows] or None


def save_feedback(conversation_id, vote):
    with connect() as conn:
        conn.execute(
            'INSERT INTO feedback (conversation_id, vote, given_at) VALUES (%s, %s, %s)',
            (conversation_id, vote, datetime.now(timezone.utc)),
        )
        conn.commit()


if __name__ == '__main__':
    init_db()
    print('tables created')

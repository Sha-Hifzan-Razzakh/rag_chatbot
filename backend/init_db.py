"""
Initialize the Postgres + pgvector database for the RAG chatbot.

What this script does:
1. Loads app settings (DATABASE_URL, PGVECTOR_COLLECTION, etc.)
2. Ensures the `vector` extension is created.
3. Initializes the PGVector collection table via get_vector_store().

Run from project root:
    python scripts/init_db.py
"""

import sys

from sqlalchemy import text

# Make sure we can import the app modules when run from project root
# (rag-chatbot/)
if __name__ == "__main__" and __package__ is None:
    # Add backend to sys.path
    import pathlib

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    BACKEND_PATH = PROJECT_ROOT / "backend"
    sys.path.insert(0, str(BACKEND_PATH))

from app.core.config import settings
from app.rag.embeddings import get_engine, get_vector_store


def ensure_vector_extension() -> None:
    """
    Ensure that the `vector` extension is created in the Postgres database.

    Note:
    - The extension must be installed in the Postgres instance (e.g. via `CREATE EXTENSION vector;`)
    - You may need superuser privileges depending on your setup.
    """
    engine = get_engine()
    print(f"Connecting to database: {settings.DATABASE_URL}")

    with engine.connect() as conn:
        print("Ensuring 'vector' extension exists...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
        print("Extension 'vector' ensured.")


def ensure_pgvector_collection() -> None:
    """
    Initialize the PGVector collection/table for embeddings.

    The langchain_postgres PGVector wrapper will create the collection/table
    when `create_collection_if_not_exists=True` (already set in embeddings.py).
    """
    print(f"Initializing PGVector collection: {settings.PGVECTOR_COLLECTION} ...")
    _ = get_vector_store()
    print("PGVector collection initialized (or already exists).")


def main() -> None:
    print("=== init_db.py: Initializing RAG Chatbot database ===")
    ensure_vector_extension()
    ensure_pgvector_collection()
    print("=== init_db.py: Done ===")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database.config import Base, engine
from backend.models.nlp_analysis import NlpAnalysis, VoiceJob, DatasetRecord, DatasetContent
from backend.models.document_record import DocumentRecord


TABLES = [
    NlpAnalysis.__table__,
    VoiceJob.__table__,
    DatasetRecord.__table__,
    DatasetContent.__table__,
    DocumentRecord.__table__,
]


def create_tables(drop_first: bool = False):
    print("KANAME [要] — Database table bootstrap")
    print(f"Engine URL: {engine.url.render_as_string(hide_password=True)}")

    if drop_first:
        print("\nDropping existing tables (drop_first=True)...")
        try:
            Base.metadata.drop_all(bind=engine, tables=TABLES)
            print("  → dropped.")
        except Exception as e:
            print(f"  ⚠ drop failed: {e}")

    print("\nCreating tables if not exists...")
    try:
        Base.metadata.create_all(bind=engine, tables=TABLES)
        print("  ✓ tables ready:")
        for t in TABLES:
            print(f"    - {t.name}")
    except Exception as e:
        print(f"  ✗ create_all failed: {e}")
        print("  Hint: Check that DATABASE_URL in .env points at a writable SQLite file (default: sqlite:///./kaname.db).")
        return 1

    print("\n✓ Database foundation initialized.")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Initialize KANAME database tables (SQLite by default)")
    parser.add_argument("--drop-first", action="store_true", help="Drop existing tables before creating")
    args = parser.parse_args()
    sys.exit(create_tables(drop_first=args.drop_first))

"""Offline demo of the sandbox gate. No LLM, no network: five migrations, five verdicts.

Run: python -m examples.sandbox_demo
"""

import os
import tempfile

from app.demo_data import make_demo_db
from app.sandbox import run_in_sandbox

FK_REBUILD = """
CREATE TABLE access_log_new (user_id INTEGER REFERENCES users(id), action TEXT, ip TEXT, at TEXT);
INSERT INTO access_log_new SELECT * FROM access_log;
DROP TABLE access_log;
ALTER TABLE access_log_new RENAME TO access_log;
"""

CASES = [
    ("add a foreign key (clean data)", "clean", FK_REBUILD),
    ("add a foreign key (one orphan row)", "orphan", FK_REBUILD),
    ("drop and recreate users (loses rows)", "clean",
     "DROP TABLE users; CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL);"),
    ("ATTACH a second database", "clean", "ATTACH DATABASE '/tmp/elsewhere.db' AS x;"),
    ("runaway query", "clean",
     "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c;"),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dbs = {"clean": os.path.join(tmp, "clean.db"), "orphan": os.path.join(tmp, "orphan.db")}
        make_demo_db(dbs["clean"], with_orphan=False)
        make_demo_db(dbs["orphan"], with_orphan=True)
        for title, which, sql in CASES:
            res = run_in_sandbox(dbs[which], sql, timeout_s=1)
            print(f"{'PASS' if res.ok else 'REFUSED':8}{title}")
            if not res.ok:
                print(f"        {res.error}")
    print("\nThe real database file was never written to.")


if __name__ == "__main__":
    main()

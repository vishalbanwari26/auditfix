"""Invented demo target database. Every value here is fake."""

import sqlite3
import sys
from pathlib import Path


def make_demo_db(path: str, with_orphan: bool = True) -> None:
    if Path(path).exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, ssn TEXT, password_plain TEXT);
        CREATE TABLE access_log (user_id INTEGER, action TEXT, ip TEXT, at TEXT);
        CREATE TABLE backups (id INTEGER PRIMARY KEY, path TEXT, encrypted INTEGER);

        INSERT INTO users VALUES
          (1, 'alice@example.test', '000-00-0001', 'not-a-real-password-1'),
          (2, 'bob@example.test',   '000-00-0002', 'not-a-real-password-2'),
          (3, 'carol@example.test', '000-00-0003', 'not-a-real-password-3');
        INSERT INTO access_log VALUES
          (1, 'login',  '192.0.2.10', '2026-01-05 09:00:00'),
          (2, 'login',  '192.0.2.11', '2026-01-05 09:05:00'),
          (2, 'export', '192.0.2.11', '2026-01-05 09:07:00');
        INSERT INTO backups VALUES
          (1, '/backups/2026-01-01.db', 1),
          (2, '/backups/2026-01-02.db', 0);
        """
    )
    if with_orphan:  # a log row for a user that does not exist: a foreign key cannot be added safely
        conn.execute("INSERT INTO access_log VALUES (9, 'login', '192.0.2.99', '2026-01-05 09:10:00')")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = args[0] if args else "target.db"
    make_demo_db(target, with_orphan="--clean" not in sys.argv)
    print(f"created {target}")

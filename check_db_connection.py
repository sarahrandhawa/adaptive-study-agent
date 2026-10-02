import psycopg
import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute('SELECT 1')
result = cur.fetchone()
print(result)
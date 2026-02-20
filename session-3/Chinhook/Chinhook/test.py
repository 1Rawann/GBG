from sqlalchemy import create_engine, text
 
 
 
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



DATABASE_URL =os.getenv("DB_URL") 
 
engine = create_engine(DATABASE_URL)
 
query = text("""
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
""")
 
 
def fetch_schema():
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            current_table = ""
            for row in result:
                table_name, column_name = row[0], row[1]
                if table_name != current_table:
                    if current_table != "":
                        print()  
                    print(f"Table: {table_name}\nColumns: ", end="")
                    current_table = table_name
                print(f"{column_name}, ", end="")
            print()
    except Exception as e:
        print(f"Error fetching schema: {e}")
 
 
if __name__ == "__main__":
    fetch_schema()
 
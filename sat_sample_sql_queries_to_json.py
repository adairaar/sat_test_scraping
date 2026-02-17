'''
Docstring for sat_sample_sql_queries_to_json

Can use function to run queries of SQL database and save results in JSON format.

Examples of SQL queries of database produced in sat_answer_key_extractor.py
Results outputted to output.json

'''

import json
import sqlite3

def export_db_to_json(query, database_file= '/outputs/sat_questions.db', json_file_path = '/outputs/output.json'):
    """Reads data from an SQLite database and exports it to a JSON file.

    Args:
        database_file (str): The path to the SQLite database file.
        query (str): The SQL SELECT query to execute.
        json_file_path (str): The path for the output JSON file.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Convert to a list of dictionaries (JSON format)
        results = [dict(zip(columns, row)) for row in rows]
        
        # Write to a JSON file
        with open(json_file_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Data successfully exported to {json_file_path}")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except IOError as e:
        print(f"File I/O error: {e}")
    finally:
        if conn:
            conn.close()


database_name = 'sat_questions.db'
sql_query = "SELECT * FROM sat_questions WHERE question_number = 1"
output_json_file = '/outputs/output0.json'
export_db_to_json(sql_query, database_name, output_json_file)

sql_query = "SELECT * FROM sat_questions WHERE question_number = 1 AND section = 'Reading and Writing'"
output_json_file = '/outputs/output1.json'
export_db_to_json(sql_query, database_name, output_json_file)
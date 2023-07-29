import sqlite3

# Connect to the database
conn = sqlite3.connect('instance\pshub1.db')

# Create a cursor
c = conn.cursor()

# Execute the query to get the schema of the client table
c.execute("PRAGMA table_info(client);")

# Fetch and print all the rows
print(c.fetchall())

# Close the connection
conn.close()

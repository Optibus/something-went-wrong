EMBEDDING_TO_DIMENSION = {
    "text-embedding-3-small": 1536,
}
embedding = "text-embedding-3-small"
index_name = "sww-demo"

# Define the columns to use for the error and solution data
SOLUTION_COLS = ["Solutions", "Issue"]
ERROR_COLS = ["Error message", "Issue", "Error title"]
import os
import argparse
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import ollama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "postgresql://user:password@localhost:5432/vectordb")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # A good starting model

class PgRagAgent:
    def __init__(self, db_conn_string, embedding_model_name, ollama_model_name):
        self.db_conn_string = db_conn_string
        self.ollama_model = ollama_model_name
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.conn = None
        print("Embedding model loaded.")

    def get_connection(self):
        """Establishes and returns a database connection."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(self.db_conn_string)
        return self.conn

    def setup_database(self):
        """Sets up the database table and pgvector extension."""
        print("Setting up database...")
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            # Get embedding dimension
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id bigserial PRIMARY KEY,
                content text,
                embedding vector({embedding_dim})
            )
            """)
            conn.commit()
        print("Database setup complete.")

    def ingest(self, directory_path):
        """Ingests all .md and .txt files from a directory into the database."""
        print(f"Starting ingestion from directory: {directory_path}")
        conn = self.get_connection()
        register_vector(conn)
        
        files_processed = 0
        for filename in os.listdir(directory_path):
            if filename.endswith((".md", ".txt")):
                filepath = os.path.join(directory_path, filename)
                print(f"  - Processing {filename}...")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Generate embedding
                    embedding = self.embedding_model.encode(content)
                    
                    # Store in database
                    with conn.cursor() as cur:
                        cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (content, embedding))
                    files_processed += 1
                except Exception as e:
                    print(f"    [!] Error processing {filename}: {e}")
        
        conn.commit()
        print(f"\nIngestion complete. Processed {files_processed} files.")

    def retrieve_context(self, query, top_k=3):
        """Retrieves the top_k most relevant document chunks for a query."""
        conn = self.get_connection()
        register_vector(conn)
        
        query_embedding = self.embedding_model.encode(query)
        
        with conn.cursor() as cur:
            # Using cosine distance for similarity search
            cur.execute("SELECT content FROM documents ORDER BY embedding <=> %s LIMIT %s", (query_embedding, top_k))
            results = [row[0] for row in cur.fetchall()]
        return results

    def query(self, user_query):
        """Performs RAG to answer a user's query."""
        print(f"\n[User Query]: {user_query}")
        print("\n1. Retrieving relevant context from database...")
        context_chunks = self.retrieve_context(user_query)
        
        if not context_chunks:
            print("\n[!] No relevant context found in the database. The LLM will answer without context.")
            context_str = "No context available."
        else:
            print(f"   - Found {len(context_chunks)} relevant chunk(s).")
            context_str = "\n\n".join(context_chunks)

        prompt = f"""
        You are a helpful AI assistant. Use the following context to answer the user's question.
        If the context does not contain the answer, state that you don't have enough information from the provided documents.

        Context:
        ---
        {context_str}
        ---

        User Question: {user_query}

        Answer:
        """

        print(f"\n2. Generating response with Ollama model '{self.ollama_model}'...")
        print("\n[AI Response]:")
        try:
            stream = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print()
        except Exception as e:
            print(f"\n[!] Error communicating with Ollama: {e}")
            print("    Please ensure Ollama is running and the model '{self.ollama_model}' is installed ('ollama pull {self.ollama_model}').")

    def close_connection(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A RAG agent using PostgreSQL with pgvector and a local LLM.")
    parser.add_argument("--setup", action="store_true", help="Initialize the database schema and pgvector extension.")
    parser.add_argument("--ingest", type=str, metavar="PATH", help="Path to a directory of .md or .txt files to ingest.")
    parser.add_argument("--query", type=str, metavar="QUESTION", help="Ask a question to the RAG agent.")
    
    args = parser.parse_args()
    
    agent = PgRagAgent(DB_CONNECTION_STRING, EMBEDDING_MODEL, OLLAMA_MODEL)

    try:
        if args.setup:
            agent.setup_database()
        elif args.ingest:
            if not os.path.isdir(args.ingest):
                print(f"[!] Error: Provided path '{args.ingest}' is not a valid directory.")
            else:
                agent.ingest(args.ingest)
        elif args.query:
            agent.query(args.query)
        else:
            print("Welcome to the pg-rag-agent! Ask a question or ingest documents.")
            print("Run with --help for options.")
            # Interactive loop if no arguments are provided
            while True:
                try:
                    user_input = input("\nEnter your question (or 'quit' to exit): ")
                    if user_input.lower() == 'quit':
                        break
                    if user_input.strip():
                        agent.query(user_input)
                except KeyboardInterrupt:
                    break
    finally:
        agent.close_connection()

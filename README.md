# PG RAG Agent: Self-Hosted Q&A with PostgreSQL & Local LLMs

This project provides a complete, self-hosted Retrieval-Augmented Generation (RAG) agent. It ingests your local documents (Markdown, text files) into a PostgreSQL database using the powerful `pgvector` extension for efficient similarity search. You can then ask questions, and the agent will retrieve relevant information from your documents to provide context-aware answers using a local LLM via Ollama.

This is a powerful and scalable way to build your own private, second brain or a question-answering system for your personal or internal documents.

![pg-rag-agent-demo](https://user-images.githubusercontent.com/12345/placeholder.gif) *<-- A conceptual demo gif would go here -->*

## Features

- **Local First**: All components can run locally, ensuring your data remains private.
- **Robust Vector Storage**: Uses PostgreSQL with `pgvector` for scalable and persistent vector storage, a production-grade alternative to file-based vector DBs.
- **Flexible LLM Support**: Integrates with any model supported by [Ollama](https://ollama.com/) (e.g., Llama 3, Mistral, Phi-3).
- **Simple CLI**: Easy-to-use command-line interface to ingest documents and ask questions.
- **Easy Setup**: Leverages Docker Compose for a one-command database setup.

## How It Works

The project follows a classic RAG pipeline:

1.  **Ingestion**: The script reads text/markdown files from a specified directory.
2.  **Embedding**: For each document, it uses a `sentence-transformers` model to generate a numerical vector (an embedding) that captures the document's semantic meaning.
3.  **Storage**: The document's content and its corresponding embedding are stored in a PostgreSQL table. The `embedding` column is of type `vector`, enabled by the `pgvector` extension.
4.  **Retrieval**: When you ask a question, your query is also converted into an embedding. The agent then performs a similarity search in the database to find the document embeddings that are closest to your query's embedding (using cosine similarity).
5.  **Augmentation & Generation**: The content of the most relevant documents (the context) is prepended to your original question in a prompt. This combined prompt is then sent to a local LLM via Ollama. The LLM uses the provided context to generate an informed and accurate answer.

## Prerequisites

1.  **Python 3.8+**
2.  **Docker and Docker Compose**: To run the PostgreSQL database.
3.  **Ollama**: Install and run Ollama. You'll also need to pull a model.

    bash
    # Install Ollama from https://ollama.com/

    # Pull a model to use (e.g., Llama 3)
    ollama pull llama3
    

## Installation

1.  **Clone the repository:**

    bash
    git clone https://github.com/bagait/pg-rag-agent.git
    cd pg-rag-agent
    

2.  **Create a virtual environment and install dependencies:**

    bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    

3.  **Setup the Database with Docker:**

    Create a `docker-compose.yml` file in the project root:

    yaml
    version: '3.8'
    services:
      db:
        image: pgvector/pgvector:pg16
        container_name: pgvector_db
        environment:
          - POSTGRES_DB=vectordb
          - POSTGRES_USER=user
          - POSTGRES_PASSWORD=password
        ports:
          - "5432:5432"
        volumes:
          - pg_data:/var/lib/postgresql/data

    volumes:
      pg_data:
    

    Then, start the database:

    bash
    docker-compose up -d
    

4.  **Configure Environment Variables:**

    Create a `.env` file in the project root. The default values match the `docker-compose.yml` file.

    env
    # .env
    DB_CONNECTION_STRING="postgresql://user:password@localhost:5432/vectordb"
    OLLAMA_MODEL="llama3"
    

5.  **Initialize the Database Schema:**

    Run the setup command to create the necessary table and enable the `vector` extension.

    bash
    python main.py --setup
    

## Usage

First, create a directory with some text files to use as your knowledge base. For example, `my_docs/`.

bash
# my_docs/project-alpha.md
Project Alpha is a new initiative focused on developing next-generation solar panels. The lead researcher is Dr. Evelyn Reed.

# my_docs/team-info.txt
Dr. Aris Thorne is the head of the material science division. He specializes in lightweight composites.


**1. Ingest your documents:**

Point the ingest command to your directory.

bash
python main.py --ingest ./my_docs/


**2. Ask a question:**

Now you can query your knowledge base.

bash
python main.py --query "Who is the lead researcher for Project Alpha?"


**Example Output:**


[User Query]: Who is the lead researcher for Project Alpha?

1. Retrieving relevant context from database...
   - Found 1 relevant chunk(s).

2. Generating response with Ollama model 'llama3'...

[AI Response]:
Based on the document provided, the lead researcher for Project Alpha is Dr. Evelyn Reed.


**3. Interactive Mode:**

To have a continuous conversation, simply run the script without any arguments.

bash
python main.py


## License

This project is licensed under the MIT License.
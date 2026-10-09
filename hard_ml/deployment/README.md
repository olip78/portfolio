# ML Service Deployment Design

*Final training project for the ML Service Deployment module of the Hard ML specialisation.*

## Task

Design a deployment architecture for a question-answering retrieval and ranking service.

## Training flow

- Generate document embeddings.
- Cluster the documents and build a vector index, such as FAISS, for each cluster.
- Train a ranking model.

## Two-stage inference

- Generate an embedding for the incoming query.
- Identify the most relevant document cluster.
- Retrieve a shortlist of candidates from the corresponding vector index.
- Rerank the shortlist with the ranking model.

```mermaid
flowchart LR
    A[Query] --> B[Query embedding]
    B --> C[Nearest cluster]
    C --> D[Vector index]
    D --> E[Candidate shortlist]
    E --> F[Ranking model]
    F --> G[Ranked results]
```

## Main design challenges

- Individual vector indexes may consume 80-90% of a server's memory.
- Model and index updates must be applied without interrupting the service.

The design therefore focuses on versioned model and index artifacts, controlled loading and unloading of large indexes, and a safe update strategy for the serving layer.

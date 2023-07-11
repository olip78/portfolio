# Final project of ML-service deployment scenarios module of Hard ML specialization

The goal of the final project is to design a prototype deployment of a QA system.

#### QA service specification:
Training:
- Document (question) embedding
- All documents (embeddings) are clustered, after that for each cluster an (e.g. FAISS) Index is calculated
- A ranking model is trained
Two step inference:
- Query embedding 
- The most relevant cluster is identified (e.g. as the nearest center)
- A shot list of candidates is selected by applying a corresponding Index
- The shot list of documents is ranked using the ranking model
#### Main chalenges:
- Each of the Indexes may well occupy 80-90% of the server memory
- A seamless update mechanism is needed
  
#### Solution: [drawio diagram](https://drive.google.com/file/d/1grCdutLZlpFe419omJgPKvxdsVqGw0ZH/view?usp=sharing)
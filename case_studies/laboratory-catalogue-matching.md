# LLM-Assisted Matching and Classification for Multilingual Laboratory Catalogues

*Commercial project, 2024 - early 2025 | Role: sole data scientist and developer*  
*Source code and client data are proprietary.*

An end-to-end, locally runnable system was developed to reduce the manual effort required to harmonise multilingual medical-laboratory catalogues. Each laboratory test was represented not as a single text field, but as a structured triple comprising an analyte, an analyser family, and a biomaterial. The source catalogues contained multiple languages, abbreviations, internal codes, inconsistent terminology, and, in some cases, insufficient information for an unambiguous match.

The matching pipeline used an LLM for translation and enrichment, embeddings for semantic search, graph-constrained candidate selection over valid test triples, and CatBoost/YetiRank to rerank the candidates. The goal was not full automation, but to provide experts with a ranked list of possible matches.

A related neural classifier separated likely laboratory tests, non-test records, and uncertain cases. It combined text embeddings with features extracted by an LLM based on questions prepared with domain experts. Noisy and inconsistent labels were improved through repeated model-assisted expert review.

Top-1 matching accuracy reached 80.3-80.9% across all catalogues and exceeded 82% on held-out data. The correct candidate appeared in the top five in approximately 94.5% of cases. Expert error analysis showed that many remaining records could not be matched without additional information; estimated accuracy on records that could be matched was approximately 96%.

The classifier achieved a cross-validated ROC AUC of 0.967. At a conservative threshold, 34.2% of non-test records could be filtered automatically with 100% precision on validation data.

Production integration into the client's core IT platform was planned as a later step.

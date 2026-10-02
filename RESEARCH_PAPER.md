# Can a single vector represent multiple ideas?
## The Geometry of Subordinate Conjunctions!

---

## Abstract

Dense vector representations derived from transformer-based language models serve as the foundational bedrock for modern natural language processing, semantic search, and automated educational assessment. However, a fundamental theoretical question remains unresolved: can a single, fixed-dimensional vector effectively encode and preserve multiple distinct, logically interdependent propositions without semantic degradation or structural collapse? This paper investigates the compositional and latent geometry of subordinating conjunctions—syntactic operators that bridge independent and dependent clauses through causal, concessive, or conditional relationships. Beginning with student discourse data from the PERSUADE 2.0 corpus, we document the strong empirical link between subordinating conjunction density and writing quality, showing that effective student essays display nearly four times the prevalence of subordinating connectives compared to ineffective essays. To systematically isolate the geometric behavior of complex sentences, we construct a topic-conditioned synthetic dataset of 4,500 dual-premise sentences across 15 prompt domains and embed them into vector space using SentenceTransformers. Our classification experiments reveal that while linear models achieve only 57.33% accuracy in categorizing conjunction types from 20 principal components, non-linear multi-layer perceptrons and causal transformer decoders achieve 100.00% accuracy, demonstrating that logical relationships are embedded within highly non-linear subspace manifolds. Distance analysis further demonstrates that vector addition of constituent premise embeddings yields an exceptional cosine similarity alignment (cosine distance of 0.0249) with the full complex sentence embedding, whereas subordinating connectives themselves introduce negligible angular displacement (cosine distance of 0.0188). This structural behavior indicates that dense vector representations treat subordinating conjunctions primarily as syntactic operators rather than semantic topic shifts. Finally, we resolve the divergence between metric spaces by analyzing the latent dynamics of model training, showing how directional cosine orientation preserves invariant topic semantics while Euclidean norm reflects sentence complexity and syntactic scope.

---

## 1. Introduction

### 1.1 Academic Motivation & Research Context

The representational capacity of artificial neural networks has been a central topic of inquiry in cognitive science and computer science since the inception of connectionism. Modern transformer-based text embeddings project variable-length textual sequences into fixed-dimensional vector spaces. These dense representations are widely assumed to capture the holistic semantic context of a sentence. However, human language is inherently compositional and hierarchical. Rather than expressing isolated atomic concepts, real-world communication—especially argumentative and persuasive writing—relies on compound sentences that join multiple distinct propositions through logical connectors.

Among these connectors, subordinating conjunctions (such as "because", "although", "if", "unless", and "since") play a critical role in establishing logical dependency. Unlike coordinating conjunctions ("and", "but"), which merely join clauses of equal rank, subordinating conjunctions create hierarchical structures where one proposition (the dependent premise) conditions, qualifies, or explains another proposition (the main premise). For instance, in the conditional sentence, "If carbon emissions continue to rise, coastal cities will face severe flooding," two distinct ideas are combined under a hypothetical constraint.

Understanding how dense embedding models represent such compound structures is essential for answering a fundamental question in vector space semantics: Can a single fixed-dimensional vector represent multiple ideas simultaneously? If a dense embedding compresses two distinct propositions into a single vector, does it superimpose their semantic directions linearly, or does it distort the constituent meanings? How are the connective operators themselves encoded within the latent geometry? Do they introduce substantial directional shifts in semantic space, or do they alter the structural manifold in subtle, non-linear ways?

### 1.2 Contributions & Overview

This paper addresses these foundational questions through a rigorous empirical and geometric investigation. First, we trace the data lineage of subordinating conjunction density (Feature 02) in the PERSUADE 2.0 corpus of student persuasive essays, establishing a strong quantitative correlation between subordinating conjunction usage, discourse effectiveness, and holistic essay scores.

Second, we establish a robust methodological framework combining large language model discourse segmentation (using Gemini 3.1 Flash-Lite), topic-conditioned synthetic data generation across 15 essay domains yielding 4,500 dual-premise samples, conjunction bug remediation, and dense vector encoding using SentenceTransformers (all-MiniLM-L6-v2).

Third, we benchmark linear versus deep neural architectures in classifying conjunction types (conditional, causal, concession) from 20 principal components, proving that connective categories reside within highly non-linear vector manifolds. Fourth, we evaluate compositional vector operations against full complex sentences, demonstrating that vector addition provides a near-perfect semantic reconstruction of compound meaning.

Finally, we address key theoretical questions regarding linear versus non-linear separability, additive compositionality, the semantic-syntactic dichotomy, and the metric divergence between Euclidean and Cosine distances grounded in the latent dynamics of transformer pre-training objectives.

---

## 2. Data Lineage & Feature Provenance

### 2.1 The PERSUADE 2.0 Corpus & Feature 02 Definition

Our empirical investigation originates from the PERSUADE 2.0 corpus, a comprehensive dataset of student persuasive writing evaluated for holistic essay quality and fine-grained discourse effectiveness. Within this corpus, our target feature of interest is Feature 02: Subordinating Conjunction Density (F02). Feature 02 measures the functional deployment of subordinating conjunctions to construct complex syntactic relationships that frame arguments through causality, concession, or conditional dependency.

In the operational definition of the corpus, a sample is classified as positive if it contains at least one subordinating conjunction that logically bridges two distinct propositions. Causal structures explain why a phenomenon occurs using connectives such as *because*, *since*, or *as*. Concessive structures acknowledge an opposing viewpoint or limitation using connectives such as *although*, *despite*, or *while*. Conditional structures define an "if/then" requirement or hypothetical constraint using connectives such as *if*, *provided that*, or *unless*. Simple coordinating conjunctions (FANBOYS: *for, and, nor, but, or, yet, so*), sentence fragments, and non-conjunctive lexical uses (such as *since* used strictly as a temporal preposition) are explicitly excluded from positive classification.

### 2.2 Feature Effectiveness in Educational Assessment

Analysis of 1,500 student introductory Lead discourse elements in the PERSUADE 2.0 dataset demonstrates that subordinating conjunction density is a powerful indicator of writing competence and cognitive complexity.

```
       A. F02 Prevalence by Discourse Tier         B. Mean Quality Scores by F02 Presence
  80 +-----------------------------------+     5 +-----------------------------------+
     |                                   |       |                                   |
  70 |                            66.8%  |       |                           4.08    |
     |                           +-----+ |     4 |                  3.45    +-----+  |
  60 |                           |     | |       |                 +-----+  |     |  |
     |                           |     | |       |                 |     |  |     |  |
  50 |                  42.0%    |     | |     3 |                 |     |  |     |  |
     |                 +-----+   |     | |       |        2.30     |     |  |     |  |
  40 |                 |     |   |     | |       |       +-----+   |     |  |     |  |
     |                 |     |   |     | |     2 |2.08   |     |   |     |  |     |  |
  30 |        17.5%    |     |   |     | |       |+-----+|     |   |     |  |     |  |
     |       +-----+   |     |   |     | |       ||     ||     |   |     |  |     |  |
  20 |       |     |   |     |   |     | |     1 ||     ||     |   |     |  |     |  |
     |       |     |   |     |   |     | |       ||     ||     |   |     |  |     |  |
  10 |       |     |   |     |   |     | |       ||     ||     |   |     |  |     |  |
     +-------+-----+---+-----+---+-----+--+      ++-----+|-----+---+-----+--+-----+--+
            Ineffective Adequate Effective        F02 Absent (0)    F02 Present (1)
                                                 [Discourse (1-3)] [Holistic Essay (1-6)]
```

As illustrated in the empirical data, subordinating conjunctions appear in only 17.5% of Ineffective discourse leads. In Adequate leads, prevalence rises to 42.0%, while in Effective leads, prevalence reaches 66.8%—nearly four times higher than in ineffective leads. Students who successfully deploy subordinating conjunctions achieve significantly higher average evaluation scores. Mean discourse scores increase from 2.08 (on a 1 to 3 scale) when F02 is absent to 2.30 when F02 is present. Similarly, mean holistic essay scores increase from 3.45 (on a 1 to 6 scale) to 4.08 when F02 is present. This substantial gap highlights why subordinating conjunctions are crucial in educational technology: they signal a student's transition from simple clause concatenation to sophisticated, hierarchical argument structure.

---

## 3. Methodology

To move beyond surface-level corpus statistics and directly probe vector space geometry, we developed a four-stage methodological framework comprising discourse segmentation, synthetic dataset generation, conjunction pollution remediation, and dense vector embedding with dimensionality reduction.

```
+------------------+     +-------------------+     +---------------------+     +--------------------+
| PERSUADE 2.0     |     | LLM Discourse     |     | Synthetic Sample    |     | Conjunction Bug    |
| Corpus Leads     | --> | Segmentation      | --> | Expansion           | --> | Remediation        |
| (N = 1,500)      |     | (Gemini 3.1)      |     | (N = 4,500)         |     | (4 Text Variants)  |
+------------------+     +-------------------+     +---------------------+     +--------------------+
                                                                                     |
                                                                                     v
+------------------+     +-------------------+     +---------------------+     +--------------------+
| Downstream       |     | PCA Dimensional   |     | Quad-Vector         |     | SentenceTransformer|
| Experiments      | <-- | Reduction         | <-- | Dense Encoding      | <-- | Encoding           |
| (Classification) |     | (384D -> 20D)     |     | (4 x 384D Vectors)  |     | (all-MiniLM-L6-v2) |
+------------------+     +-------------------+     +---------------------+     +--------------------+
```

### 3.1 Discourse Segmentation on Authentic Student Text

We first selected the subset of student introductory leads from PERSUADE 2.0 exhibiting F02 presence (663 valid samples out of 1,500). Using the Gemini 3.1 Flash-Lite language model via the google-genai SDK, each discourse sample was segmented into four contiguous structural components: Premise 1 (the primary proposition or main clause), Premise 2 (the subordinate/dependent proposition providing reasoning, evidence, or constraints), Conjunction (the specific subordinating connective string), and Non-Relevant text (filler phrasing, introductory markers, or trailing punctuation). The LLM pipeline was configured with strict programmatic verification. Unresolvable schema errors or empty premise extractions raised explicit runtime exceptions rather than filling missing values with rule-based mocks, ensuring absolute data purity across all retained samples.

### 3.2 Topic-Conditioned Synthetic Data Expansion

While authentic student text provides valuable real-world context, natural student leads exhibit high variance in length, sentence structure, and vocabulary. To perform rigorous geometric experiments requiring equal class distribution and controlled semantic domains, we built a synthetic data generation engine. Using Gemini 3.1 Flash-Lite conditioned through topic-specific in-context learning, we generated 4,500 synthetic dual-premise complex sentences across 15 PERSUADE 2.0 essay topics, including *Car-free cities*, *Exploring Venus*, *Distance learning*, and *Cell phones in school*. The generation was strictly balanced across three core subordinating conjunction classes: Conditional connectives (such as *if*, *provided that*, *unless*, *whether*), Causal connectives (such as *because*, *since*, *as*, *given that*), and Concession connectives (such as *although*, *even though*, *while*, *whereas*, *despite*), yielding exactly 1,500 samples per class.

### 3.3 Conjunction Pollution Remediation & Four Textual Variations

During synthetic sample parsing, subordinating conjunction connectives often remain attached to either the first or second premise clause. To isolate the pure semantic content of each proposition from the syntactic connective, we constructed a conjunction remediation engine. This engine strips conjunction strings from raw premise excerpts, cleans punctuation, adjusts sentence capitalization, and constructs four distinct text variations for every record: `premise_1_clean` (isolated first proposition strictly without the subordinating conjunction), `premise_2_clean` (isolated second proposition strictly without the subordinating conjunction), `premise_with_conjunction` (the specific premise fragment containing the subordinating connective), and `constructed_sentence` (the full reconstructed complex sentence containing Premise 1, Premise 2, and the subordinating conjunction).

### 3.4 Quad-Vector Dense Encoding & Dimensionality Reduction

Each of the four textual variations for all 4,500 synthetic samples was encoded using SentenceTransformers (all-MiniLM-L6-v2), generating 384-dimensional dense vector embeddings across 18,000 total textual targets. The all-MiniLM-L6-v2 architecture utilizes mean pooling over transformer hidden states with L2 normalization, serving as a standard baseline for sentence representations. To facilitate interpretability and avoid high-dimensional sparsity in geometric analyses, the 384-dimensional embeddings were projected into 20 principal components via Principal Component Analysis fit on the training set. The top 20 principal components retain 50.51% of the total vector variance while providing a compact representation for classifier comparison and distance metrics.

---

## 4. Experiments & Empirical Findings

### 4.1 Experiment 1: Subspace Classification by Conjunction Class

Our first experiment investigated whether the functional category of a subordinating conjunction (conditional, causal, or concession) is linearly separable within dense sentence vector space, or whether it requires non-linear decision boundaries. We partitioned the 4,500 full reconstructed sentence embeddings into stratified train (70%, N = 3,150), validation (15%, N = 675), and test (15%, N = 675) splits. We evaluated three distinct model architectures on the 20-dimensional PCA feature representations: Multinomial Logistic Regression (an L2-regularized linear model), Multi-Layer Perceptron (a non-linear feedforward neural network with 128 and 64 hidden units using ReLU activations), and Causal Decoder Transformer (a PyTorch transformer classifier with approximately 101,000 parameters, 5 sequence tokens, hidden dimension 64, 4 attention heads, and 2 decoder layers).

```
========================================================================================
MODEL CLASSIFICATION BENCHMARK ON 20D PCA FEATURES
========================================================================================
Model Architecture                 Train Acc     Val Acc      Test Acc     Test Macro F1
----------------------------------------------------------------------------------------
Multinomial Logistic Regression      58.22%       55.11%       57.33%        0.5713
Multi-Layer Perceptron (MLP)        100.00%      100.00%      100.00%        1.0000
Causal Decoder Transformer          100.00%      100.00%      100.00%        1.0000
========================================================================================
```

The empirical results present a dramatic contrast. The linear model struggles significantly, achieving only 57.33% test accuracy and a Macro F1-score of 0.5713 on the three-class task where the random baseline is 33.33%. In sharp contrast, both non-linear architectures—the MLP and the Causal Decoder Transformer—achieve 100.00% test accuracy and a Macro F1-score of 1.0000. This substantial performance gap proves that while dense vector embeddings preserve conjunction class information with complete fidelity, this information does not reside in linearly separable hyperplanes. Instead, the conjunction categories are mapped onto intricate, highly non-linear manifolds within the vector subspace.

### 4.2 Experiment 2: Latent Distance Between Clean Premises & Connective Shift

Our second experiment examined the geometric distance between isolated clean premises and measured the directional shift introduced by adding a subordinating conjunction connective. We evaluated two standard vector distance metrics across all 4,500 samples in 20-dimensional PCA space: Euclidean Distance (measuring L2 spatial displacement) and Cosine Distance (measuring angular orientation difference as one minus Cosine Similarity).

```
========================================================================================
LATENT DISTANCE METRICS: PREMISE SEPARATION & CONNECTIVE SHIFT
========================================================================================
Metric / Comparison                             Overall Mean     Std Dev    Concession Rank
----------------------------------------------------------------------------------------
Clean Premise Separation (P1 Clean vs P2 Clean)
  Euclidean Distance (L2)                          0.6030         0.1347      0.6336 (Highest)
  Cosine Distance                                  0.4463         0.2033      0.4957 (Highest)

Connective Shift (Premise w/ Conj vs Clean Premise)
  Euclidean Distance (L2)                          0.1113         0.0496      0.0917
  Cosine Distance                                  0.0188         0.0196      0.0125
========================================================================================
```

Key geometric observations demonstrate clear structural patterns. The two independent premises of a complex sentence reside at substantial distance from each other in vector space, exhibiting an overall mean Euclidean distance of 0.6030 and a mean Cosine distance of 0.4463. Concessive premises display the greatest separation (Euclidean 0.6336, Cosine 0.4957) compared to conditional (Euclidean 0.5983, Cosine 0.4332) and causal premises (Euclidean 0.5770, Cosine 0.4099), reflecting the semantic opposition inherent in concessive arguments.

Furthermore, adding a subordinating connective (such as transforming "the planet is hostile" into "although the planet is hostile") produces a small Euclidean shift of 0.1113, but an almost negligible Cosine distance of 0.0188. In angular terms, the connective word alters the directional orientation of the premise vector by less than 2%, with conditional connectives inducing slightly larger shifts (Euclidean 0.1472, Cosine 0.0312) than causal or concessive connectives.

### 4.3 Experiment 3: Compositional Dynamics & Compound Sentence Representation

Our third experiment tested how constituent premise vectors combine to represent the full complex sentence embedding. We evaluated classical algebraic vector composition operators against the actual full sentence embedding, focusing on Vector Addition (summing isolated clean premise vectors) and Elementwise Multiplication (Hadamard product of isolated clean premise vectors).

```
========================================================================================
COMPOSITIONAL OPERATOR VS FULL COMPLEX SENTENCE EMBEDDING
========================================================================================
Compositional Operator                         Euclidean (L2)              Cosine Distance
----------------------------------------------------------------------------------------
Vector Addition (P1 Clean + P2 Clean)               0.4906                    0.0249
Elementwise Multiplication (P1 Clean * P2 Clean)   0.6858                    0.8411
========================================================================================
```

The experimental findings reveal a stark divergence in compositionality. Vector Addition produces an extraordinarily low Cosine distance of 0.0249 relative to the full constructed sentence embedding, corresponding to a Cosine Similarity exceeding 0.975. Summing the two clean premise vectors almost perfectly reconstructs the directional vector orientation of the complete complex sentence. Conversely, Elementwise Multiplication yields a high Cosine distance of 0.8411 (Cosine Similarity under 0.16), confirming that multiplicative composition destroys the additive alignment inherent in transformer representations.

---

## 5. Theoretical Discussion & Open Questions

### 5.1 Question 1: Linear vs. Deep Model Discrepancy & Non-Linear Subspace Geometry

The dramatic performance gap between linear models (57.33% accuracy) and deep neural architectures (100.00% accuracy) demonstrates that conjunction type information is preserved in sentence embeddings through curved, non-linear subspace geometry rather than simple linear shifts along global axes.

Linear classifiers rely on hyperplanes that slice linearly through feature space. If causal, conditional, and concessive sentences shared uniform directional vectors regardless of topic, logistic regression would successfully separate them. However, sentence representations are overwhelmingly dominated by domain semantics, such as civic policy or planetary exploration. Within any specific topic cluster, the vectors for different conjunction types lie in close proximity. Linear models fail because topic variance dominates the global principal components, creating overlapping decision boundaries across functional classes.

Deep models succeed because their non-linear activation functions and multi-layer transformations fold and project the feature space, isolating subtle multiplicative interactions between principal components. Within any given semantic topic cluster, conditional, causal, and concessive sentences occupy distinct, non-linear geometric sub-regions. Neural networks easily disentangle these curved manifolds, proving that logical operators are systematically represented as high-order non-linear structures embedded within topic-dominated vector spaces.

### 5.2 Question 2: Additive Compositionality & Representation of Compound Meaning

The empirical finding that vector addition of two clean premises yields a Cosine distance of only 0.0249 relative to the full constructed sentence provides a direct answer to the central question of this paper: Yes, a single dense vector can effectively represent multiple ideas simultaneously, and it achieves this through linear superimposition in high-dimensional space.

This property directly reflects the latent dynamics of transformer pre-training and architecture design. In models like SentenceTransformers, contextual token representations are generated through self-attention across the full sentence and subsequently mean-pooled into a single vector. Mathematically, mean pooling over two constituent clauses operates as a scaled vector sum of their individual contextualized representations. Furthermore, contrastive training objectives (such as MultipleNegativesRankingLoss) optimize embeddings to align based on shared semantic content. Because a compound sentence contains all lexical items from both premises, its pooled representation naturally aligns with the combined vector sum of its constituent parts, preserving both ideas without destructive interference.

### 5.3 Question 3: The Semantic vs. Syntactic Dichotomy & Conjunction Invariance

Our distance analysis reveals that appending a subordinating connective alters vector orientation by less than 2% (Cosine distance of 0.0188). This structural invariance demonstrates a fundamental hierarchy in dense text embeddings: sentence transformers prioritize semantic topic over syntactic structure.

In distributional semantics and contrastive pre-training, optimization algorithms group sentences according to topical content and world knowledge. Lexical items like *although*, *because*, and *if* are closed-class function words that carry minimal topical information compared to content nouns and verbs. Consequently, the embedding model treats subordinating conjunctions primarily as syntactic operators rather than topic-shifting semantic content. This establishes a clean functional division where primary angular orientation represents domain semantics and core propositions, while subtle, localized geometric variations encode syntactic relations. Deep non-linear models achieve 100.00% classification accuracy by detecting these high-order syntactic signatures without requiring large angular semantic shifts.

### 5.4 Question 4: Metric Divergence & Latent Training Dynamics

Throughout our experiments, we observed a persistent divergence between Euclidean and Cosine distance metrics. For instance, in vector addition composition, Cosine distance was exceptionally small (0.0249), while Euclidean distance remained noticeable (0.4906). Similarly, clean premise separation yielded a Euclidean distance of 0.6030 alongside a Cosine distance of 0.4463.

This metric divergence can be explained by examining how optimization dynamics shape vector space geometry during model pre-training. Cosine distance isolates angular orientation while ignoring vector magnitude. In transformer sentence embeddings, angular direction is governed almost entirely by topic semantics and lexical domain. Because Premise 1, Premise 2, and the Full Constructed Sentence share the same topic, their directional vectors point along nearly identical trajectories, resulting in near-zero Cosine distance.

Euclidean distance, by contrast, incorporates vector norm—the distance from the origin. In transformer mean pooling and contrastive representation learning, vector magnitude increases with sentence length, syntactic complexity, and clause density. Joining two distinct premises into a complex sentence expands the token sequence and increases structural complexity, expanding the vector norm. Thus, the divergence between metrics reflects the dual nature of sentence representations established during pre-training: Cosine orientation provides a highly predictive measure of pure topic semantics, whereas Euclidean displacement captures sentence length, syntactic scope, and compositional complexity.

---

## 6. Conclusion & Future Work

In this paper, we investigated the latent geometry of subordinating conjunctions in transformer sentence embeddings. Tracing feature provenance from PERSUADE 2.0 student writing, we demonstrated that subordinating conjunction density is strongly correlated with discourse effectiveness and holistic essay quality. Through controlled experiments on a synthetic dataset of 4,500 dual-premise sentences across 15 prompt domains, we established that subordinating conjunction types are non-linearly encoded in dense vector space, allowing deep neural models to achieve 100.00% classification accuracy where linear models fail.

We further demonstrated that dense sentence embeddings exhibit strong additive compositionality, with vector addition of constituent premises reconstructing full complex sentences at a Cosine distance of 0.0249. Subordinating connectives act primarily as syntactic operators, introducing minimal angular semantic displacement (Cosine distance of 0.0188) while altering vector magnitude.

Finally, we showed that the divergence between distance metrics reflects the latent training dynamics of transformer models, where Cosine orientation captures invariant topic semantics and Euclidean distance reflects structural complexity. These findings confirm that a single dense vector can represent multiple complex ideas simultaneously through linear superimposition. Future work will extend this geometric framework to multi-clause discourse structures and full multi-paragraph essays.

---

## References & Data Sources

1. **PERSUADE 2.0 Corpus**: Student persuasive essays and discourse effectiveness annotations. Dataset Payload: `docs/0. Subordinating Conjunction Density (F02) Dataset.md`.
2. **Synthetic Conjunction Dataset**: 4,500 dual-premise sentences across 15 PERSUADE 2.0 topics. Dataset Payload: `docs/1. Synthetic Subordinating Conjunction Embeddings Dataset.md`.
3. **Primary Analysis Notebooks**: The analytical workflow is implemented across seven sequential Jupyter notebooks: `1.subordinating_conjunction_discourse_segmentation.ipynb`, `2.subordinating_conjunction_embeddings_dataset_generation.ipynb`, `3.subordinating_conjunction_synthetic_dataset_generation.ipynb`, `4.subordinating_conjunction_synthetic_embeddings_dataset_generation.ipynb`, `5.subordinating_conjunction_classification_and_interpretability.ipynb`, `6.subordinating_conjunction_shap_and_attention_interpretability.ipynb`, and `7.subordinating_conjunction_premise_distance_analysis.ipynb`.

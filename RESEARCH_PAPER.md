# Can a single vector represent multiple ideas?
## The Geometry of Subordinate Conjunctions!

---

## Abstract

Dense vector representations derived from transformer-based language models serve as the foundational bedrock for modern natural language processing, semantic search, and automated educational assessment. However, a fundamental theoretical question remains unresolved: can a single, fixed-dimensional vector effectively encode and preserve multiple distinct, logically interdependent propositions without semantic degradation or structural collapse? This paper investigates the compositional and latent geometry of subordinating conjunctions—syntactic operators that bridge independent and dependent clauses through causal, concessive, or conditional relationships. Beginning with student discourse data from the PERSUADE 2.0 corpus, we document the strong empirical link between subordinating conjunction density and writing quality, showing that effective student essays display nearly four times the prevalence of subordinating connectives compared to ineffective essays. To systematically isolate the geometric behavior of complex sentences, we construct a topic-conditioned synthetic dataset of 4,500 dual-premise sentences across 15 prompt domains and embed them into vector space using SentenceTransformers. Our classification experiments reveal that while linear models achieve only 57.33% accuracy in categorizing conjunction types from 20 principal components, non-linear multi-layer perceptrons and causal transformer decoders achieve 100.00% accuracy, demonstrating that logical relationships are embedded within highly non-linear subspace manifolds. Distance analysis further reveals that vector addition of constituent premise embeddings yields an exceptional cosine similarity alignment (cosine distance of 0.0249) with the full complex sentence embedding, whereas elementwise multiplication fails (cosine distance of 0.8411). Furthermore, subordinating connectives themselves introduce minimal angular shift (cosine distance of 0.0188), proving that dense embeddings treat subordinating conjunctions primarily as syntactic operators rather than semantic topic shifts. Finally, we resolve the divergence between Euclidean and Cosine metrics by proposing a hyperbolic geometry model for sentence representations, where Cosine distance encodes invariant semantic orientation and Euclidean magnitude reflects syntactic complexity and sentence scope.

---

## 1. Introduction

### 1.1 Academic Motivation & Research Context

The representational capacity of artificial neural networks has been a central topic of inquiry in cognitive science and computer science since the inception of connectionism. Modern transformer-based text embeddings project variable-length textual sequences into fixed-dimensional vector spaces. These dense representations are widely assumed to capture the holistic semantic context of a sentence. However, human language is inherently compositional and hierarchical. Rather than expressing isolated atomic concepts, real-world communication—especially argumentative and persuasive writing—relies on compound sentences that join multiple distinct propositions through logical connectors.

Among these connectors, subordinating conjunctions (such as "because", "although", "if", "unless", and "since") play a critical role in establishing logical dependency. Unlike coordinating conjunctions ("and", "but"), which merely join clauses of equal rank, subordinating conjunctions create hierarchical structures where one proposition (the dependent premise) conditions, qualifies, or explains another proposition (the main premise). For instance, in the conditional sentence, "If carbon emissions continue to rise, coastal cities will face severe flooding," two distinct ideas are combined under a hypothetical constraint.

Understanding how dense embedding models represent such compound structures is essential for answering a fundamental question in vector space semantics: **Can a single fixed-dimensional vector represent multiple ideas simultaneously?**

If a dense embedding compresses two distinct propositions into a single vector, does it superimpose their semantic directions linearly, or does it distort the constituent meanings? How are the connective operators themselves encoded within the latent geometry? Do they introduce substantial directional shifts in semantic space, or do they alter the structural manifold in subtle, non-linear ways?

### 1.2 Contributions & Overview

This paper addresses these foundational questions through a rigorous empirical and geometric investigation. The key contributions of this work are as follows:

1. **Empirical Educational Validation**: We trace the data lineage of subordinating conjunction density (Feature 02) in the PERSUADE 2.0 corpus of student persuasive essays, establishing a strong correlation between subordinating conjunction usage, discourse effectiveness, and holistic essay scores.
2. **Methodological Framework for Discourse Geometry**: We establish a robust pipeline combining large language model discourse segmentation (using Gemini 3.1 Flash-Lite), topic-conditioned synthetic data generation across 15 essay domains (4,500 dual-premise samples), conjunction bug remediation, and dense vector encoding using SentenceTransformers (all-MiniLM-L6-v2).
3. **Subspace Classification & Non-Linear Manifold Analysis**: We benchmark linear vs. deep neural models in classifying conjunction types (conditional, causal, concession) from 20 principal components, proving that connective categories are encoded in non-linear vector manifolds.
4. **Compositional Geometry & Vector Operations**: We evaluate vector addition and elementwise multiplication against full complex sentences, demonstrating that vector addition provides a near-perfect semantic reconstruction of compound meaning.
5. **Theoretical Resolution of Latent Geometry**: We discuss four fundamental open questions regarding linear vs. non-linear separability, additive compositionality, the semantic-syntactic dichotomy, and the divergence between Euclidean and Cosine distances under a proposed hyperbolic vector space model.

---

## 2. Data Lineage & Feature Provenance

### 2.1 The PERSUADE 2.0 Corpus & Feature 02 Definition

Our empirical investigation originates from the PERSUADE 2.0 corpus, a comprehensive dataset of student persuasive writing evaluated for holistic essay quality and fine-grained discourse effectiveness. Within this corpus, our target feature of interest is **Feature 02: Subordinating Conjunction Density (F02)**.

Feature 02 measures the functional deployment of subordinating conjunctions to construct complex syntactic relationships that frame arguments through causality, concession, or conditional dependency. In the corpus operational definition, a sample is classified as positive (F02 = 1) if it contains at least one subordinating conjunction that logically bridges two distinct propositions:
- **Causality**: Explaining why a phenomenon occurs using connectives such as *because*, *since*, or *as*.
- **Concession**: Acknowledging an opposing viewpoint or limitation using connectives such as *although*, *despite*, or *while*.
- **Conditional Constraints**: Defining an "if/then" requirement or hypothetical constraint using connectives such as *if*, *provided that*, or *unless*.

Simple coordinating conjunctions (FANBOYS: *for, and, nor, but, or, yet, so*), sentence fragments, and non-conjunctive lexical uses (e.g., *since* used strictly as a temporal preposition) are explicitly excluded from positive F02 classification.

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

As illustrated in the experimental data:
- **Prevalence across Performance Tiers**: Subordinating conjunctions appear in only 17.5% of *Ineffective* discourse leads. In *Adequate* leads, prevalence rises to 42.0%. In *Effective* leads, prevalence reaches 66.8%—nearly four times higher than in ineffective leads.
- **Impact on Quality Scores**: Students who successfully deploy subordinating conjunctions achieve significantly higher average evaluation scores. Mean discourse scores increase from 2.08 (on a 1 to 3 scale) when F02 is absent to 2.30 when F02 is present. Similarly, mean holistic essay scores increase from 3.45 (on a 1 to 6 scale) to 4.08 when F02 is present.

This substantial gap highlights why subordinating conjunctions are crucial in educational technology: they signal a student's transition from simple clause concatenation to sophisticated, hierarchical argument structure.

---

## 3. Methodology

To move beyond surface-level corpus statistics and directly probe vector space geometry, we developed a four-stage methodological framework: discourse segmentation, synthetic dataset generation, conjunction pollution remediation, and dense vector embedding with dimensionality reduction.

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

We first selected the subset of student introductory leads from PERSUADE 2.0 exhibiting F02 presence (663 valid samples out of 1,500). Using the Gemini 3.1 Flash-Lite language model via the google-genai SDK, each discourse sample was segmented into four contiguous structural components:
1. `premise_1`: The primary proposition or main clause.
2. `premise_2`: The subordinate/dependent proposition providing reasoning, evidence, or constraints.
3. `conjunction`: The specific subordinating connective string.
4. `non_relevant`: Any filler text, introductory phrases, or trailing punctuation.

The LLM pipeline was configured with strict programmatic verification. Unresolvable schema errors or empty premise Extractions raised explicit runtime exceptions rather than filling missing values with rule-based mocks, ensuring absolute data purity across all retained samples.

### 3.2 Topic-Conditioned Synthetic Data Expansion

While authentic student text provides valuable real-world context, natural student leads exhibit high variance in length, sentence structure, and vocabulary. To perform rigorous geometric experiments requiring equal class distribution and controlled semantic domains, we built a synthetic data generation engine.

Using Gemini 3.1 Flash-Lite conditioned through topic-specific in-context learning, we generated N = 4,500 synthetic dual-premise complex sentences across 15 PERSUADE 2.0 essay topics (e.g., *Car-free cities*, *Exploring Venus*, *Distance learning*, *Cell phones in school*). The generation was strictly balanced across three core subordinating conjunction classes:
- **Conditional** (1,500 samples): Connectives including *if*, *provided that*, *unless*, *whether*.
- **Causal** (1,500 samples): Connectives including *because*, *since*, *as*, *given that*.
- **Concession** (1,500 samples): Connectives including *although*, *even though*, *while*, *whereas*, *despite*.

### 3.3 Conjunction Pollution Remediation & Four Textual Variations

During synthetic sample parsing, subordinating conjunction connectives often remain attached to either the first or second premise clause. To isolate the pure semantic content of each proposition from the syntactic connective, we constructed a conjunction remediation engine.

This engine strips conjunction strings from raw premise excerpts, cleans punctuation, adjusts sentence capitalization, and constructs four distinct text variations for every record:
1. `premise_1_clean`: Isolated first proposition strictly without the subordinating conjunction.
2. `premise_2_clean`: Isolated second proposition strictly without the subordinating conjunction.
3. `premise_with_conjunction`: The specific premise fragment containing the subordinating connective (e.g., "because the planet is hostile").
4. `constructed_sentence`: The full reconstructed complex sentence containing Premise 1, Premise 2, and the subordinating conjunction.

### 3.4 Quad-Vector Dense Encoding & Dimensionality Reduction

Each of the four textual variations for all 4,500 synthetic samples was encoded using SentenceTransformers (`all-MiniLM-L6-v2`), generating 384-dimensional dense vector embeddings (18,000 total vectors across the dataset). The `all-MiniLM-L6-v2` model uses mean pooling over transformer hidden states with L2 normalization, making it a standard benchmark for sentence vector representations.

To facilitate interpretability and avoid the curse of dimensionality in geometric analyses, the 384-dimensional embeddings were projected into 20 principal dimensions using Principal Component Analysis (PCA) fit on the training set. The top 20 principal components retain 50.51% of the total vector variance while providing a compact representation for classifier comparison and distance metrics.

---

## 4. Experiments & Empirical Findings

### 4.1 Experiment 1: Subspace Classification by Conjunction Class

Our first experiment investigated whether the functional category of a subordinating conjunction (conditional, causal, or concession) is linearly separable within dense sentence vector space, or whether it requires non-linear decision boundaries.

We partitioned the 4,500 full reconstructed sentence embeddings into stratified train (70%, N = 3,150), validation (15%, N = 675), and test (15%, N = 675) splits. We evaluated three distinct model architectures on the 20-dimensional PCA feature representations:
1. **Multinomial Logistic Regression**: An L2-regularized linear model.
2. **Multi-Layer Perceptron (MLP)**: A non-linear feedforward neural network (128 hidden units -> 64 hidden units with ReLU activations).
3. **Causal Decoder Transformer**: A PyTorch transformer classifier (~101,000 parameters, 5 sequence tokens, hidden dimension 64, 4 attention heads, 2 decoder layers).

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

The empirical results present a dramatic contrast:
- The linear model struggles significantly, achieving only **57.33% test accuracy** and a Macro F1-score of 0.5713 on the three-class task (where random baseline is 33.33%).
- In sharp contrast, both non-linear architectures—the MLP and the Causal Decoder Transformer—achieve **100.00% test accuracy** and a Macro F1-score of 1.0000.

This huge performance gap proves that while dense vector embeddings preserve conjunction class information with complete fidelity, this information does not reside in linearly separable hyperplanes. Instead, the conjunction categories are mapped onto intricate, highly non-linear manifolds within the vector subspace.

### 4.2 Experiment 2: Latent Distance Between Clean Premises & Connective Shift

Our second experiment examined the geometric distance between isolated clean premises and measured the directional shift introduced by adding a subordinating conjunction connective.

We evaluated two standard vector distance metrics across all 4,500 samples in 20-dimensional PCA space:
- **Euclidean Distance**: Measuring L2 spatial displacement.
- **Cosine Distance**: Measuring angular orientation difference (1 minus Cosine Similarity).

```
========================================================================================
LATENT DISTANCE METRICS: PREMISE SEPARATION & CONNECTIVE SHIFT
========================================================================================
Metric / Comparison                             Overall Mean     Std Dev    Concession Rank
----------------------------------------------------------------------------------------
Clean Premise Separation (P1 Clean vs P2 Clean)
  - Euclidean Distance (L2)                        0.6030         0.1347      0.6336 (Highest)
  - Cosine Distance                                0.4463         0.2033      0.4957 (Highest)

Connective Shift (Premise w/ Conj vs Clean Premise)
  - Euclidean Distance (L2)                        0.1113         0.0496      0.0917
  - Cosine Distance                                0.0188         0.0196      0.0125
========================================================================================
```

Key geometric observations include:
1. **Clean Premise Separation**: The two independent premises of a complex sentence reside at substantial distance from each other in vector space (mean Euclidean distance 0.6030, mean Cosine distance 0.4463). Concessive premises display the greatest separation (Euclidean 0.6336, Cosine 0.4957) compared to conditional (Euclidean 0.5983, Cosine 0.4332) and causal premises (Euclidean 0.5770, Cosine 0.4099), reflecting the semantic opposition inherent in concessive arguments.
2. **Subordinating Connective Shift**: Adding a subordinating connective (e.g., transforming "the planet is hostile" into "although the planet is hostile") produces a small Euclidean shift of 0.1113, but an almost negligible Cosine distance of **0.0188**. In angular terms, the connective word alters the directional orientation of the premise vector by less than 2%. Conditional connectives induce slightly larger shifts (Euclidean 0.1472, Cosine 0.0312) than causal or concessive connectives.

### 4.3 Experiment 3: Compositional Dynamics & Compound Sentence Representation

Our third experiment tested how constituent premise vectors combine to represent the full complex sentence embedding. We evaluated two classical algebraic vector composition operators against the actual full sentence embedding:
1. **Vector Addition**: Summing the isolated clean premise vectors (Premise 1 Clean + Premise 2 Clean).
2. **Elementwise Multiplication**: Hadamard product of the isolated clean premise vectors (Premise 1 Clean multiplied elementwise with Premise 2 Clean).

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

The experimental findings reveal a stark divergence in compositionality:
- **Vector Addition**: Produces an extraordinarily low Cosine distance of **0.0249** relative to the full constructed sentence embedding. This corresponds to a Cosine Similarity exceeding 0.975! Summing the two clean premise vectors almost perfectly reconstructs the directional vector orientation of the complete complex sentence.
- **Elementwise Multiplication**: Fails entirely as a compositional operator, producing a huge Cosine distance of **0.8411** (Cosine Similarity under 0.16).

---

## 5. Theoretical Discussion & Open Questions

The empirical findings from our three experiments raise four profound theoretical questions concerning vector space semantics, model training objectives, and sentence geometry.

### 5.1 Question 1: Linear vs. Deep Model Discrepancy & Non-Linear Subspace Geometry

*Why do linear models fail (57.33% accuracy) while deep non-linear models achieve perfect prediction (100.00% accuracy)? What is the underlying geometry of conjunction types?*

The contrast between linear and non-linear classification performance demonstrates that conjunction type information is present in sentence embeddings, but it is not stored as a simple linear shift along fixed feature axes.

In linear models, prediction relies on hyperplanes that slice linearly through feature space. If causal, conditional, and concessive sentences shared uniform directional vectors regardless of topic, logistic regression would successfully separate them. However, because sentence embeddings are strongly dominated by topic semantics (such as *car-free cities* or *space exploration*), the representations for different conjunction types within the same topic cluster close together.

Deep models (MLPs and Transformers) succeed because they learn non-linear feature interactions and manifold transformations. Rather than measuring global vector direction, non-linear layers fold and project the 20-dimensional subspace, isolating subtle multiplicative interactions between principal components. The geometry of subordinating conjunctions is thus a **curved manifold structure**: within any topic cluster, conditional, causal, and concessive sentences occupy distinct, non-linear geometric sub-regions. Deep architectures easily untangle these curved boundaries, whereas linear models are unable to resolve them without overlapping decision regions.

### 5.2 Question 2: Additive Compositionality & Representation of Compound Meaning

*Why is vector addition so predictive (Cosine distance 0.0249) when comparing constituent premises to the full complex sentence? What does this reveal about embedding capacity?*

The finding that vector addition of two clean premises yields a Cosine distance of only 0.0249 relative to the full sentence provides a direct answer to the title of this paper: **Yes, a single vector can represent multiple ideas, and it does so through linear superimposition in high-dimensional space.**

This phenomenon is a direct consequence of how sentence transformer models are trained:
1. **Transformer Self-Attention & Mean Pooling**: In models like `all-MiniLM-L6-v2`, contextual token representations are computed across the entire sentence and subsequently averaged (mean pooled) to produce the final sentence vector. Mathematically, mean pooling over two constituent clauses behaves as a scaled vector sum of their individual contextualized representations.
2. **Contrastive Objective Alignment**: Contrastive training objectives (such as MultipleNegativesRankingLoss) force sentence vectors to align based on shared semantic content. Because a compound sentence contains all lexical items from both Premise 1 and Premise 2, its representation naturally aligns with the combined vector sum of its constituent parts.

This additive compositionality demonstrates that high-dimensional vector spaces possess sufficient capacity to superimpose multiple semantic propositions without destructive interference. Provided the constituent premises occupy distinct semantic directions, their vector sum preserves both ideas simultaneously within the single dense embedding.

### 5.3 Question 3: The Semantic vs. Syntactic Dichotomy & Conjunction Invariance

*Why do subordinating conjunctions introduce almost no directional distance (Cosine distance 0.0188)? What does this reveal about semantic vs. syntactic representations in embeddings?*

Our distance analysis revealed that appending a subordinating connective (transforming "the planet is hostile" into "although the planet is hostile") alters the Cosine orientation by less than 2%. This reveals a fundamental property of dense sentence embeddings: **they prioritize semantic topic over syntactic structure.**

In distributional semantics and contrastive pre-training, models are optimized to group sentences that share topical meaning and world knowledge. Lexical items like *although*, *because*, and *if* are closed-class function words that carry minimal topical information compared to content nouns and verbs (*planet*, *hostile*, *flooding*). As a result, the embedding algorithm treats subordinating conjunctions primarily as syntactic operators rather than topic-shifting content.

This creates an interesting theoretical distinction:
- **Semantics**: Represented by the primary angular direction (Cosine orientation) of the vector, which is governed by core propositions and topic content.
- **Syntax**: Represented by subtle, localized geometric shifts that do not alter the main topic direction, but can be detected with 100% accuracy by deep non-linear classifiers.

If conjunctions do not alter the primary semantic vector direction, how can deep models classify them perfectly? Deep models do not rely on large angular shifts; instead, they detect localized, high-order polynomial relationships across minor principal components that encode syntactic function words without disturbing the primary semantic topic vector.

### 5.4 Question 4: Distance Metric Divergence & Hyperbolic Sentence Geometry

*Why does Euclidean distance show noticeable variation while Cosine distance remains extremely small? Is there evidence of an underlying hyperbolic geometry in sentence embeddings?*

Throughout our experiments, we observed a striking divergence between Euclidean and Cosine metrics. For example, in vector addition composition, the Cosine distance was virtually zero (0.0249), while the Euclidean distance remained noticeable (0.4906). Similarly, clean premise separation yielded a Euclidean distance of 0.6030 alongside a Cosine distance of 0.4463.

This persistent metric divergence can be explained by an **underlying hyperbolic or hierarchical geometry** in dense sentence embeddings:

```
                      HYPERBOLIC CONE OF SENTENCE EMBEDDINGS

                                Origin (0,0,0)
                                      /\
                                     /  \
                                    /    \      Vector Norm ||V||
                                   /      \     (Syntactic Complexity
                                  /  P1    \     & Sentence Length)
                                 /    \     \
                                /      \ P2  \
                               /________\____\
                              /   Full Sentence \
                             +-------------------+
                             Angular Orientation
                             (Pure Topic Semantics)
```

1. **Cosine Distance as Angular Semantic Orientation**: Cosine distance measures the angle between vectors, completely ignoring vector magnitude. Angular orientation encodes pure semantic topic content (e.g., whether a sentence discusses space exploration or public health policy). Because Premise 1, Premise 2, and the Full Sentence all discuss the same topic, their vectors point in nearly identical directions, resulting in extremely low Cosine distance.
2. **Euclidean Distance as Vector Magnitude & Syntactic Complexity**: Euclidean distance incorporates vector norm (length from origin). In transformer mean pooling, vector norm increases with text length, lexical specificity, and syntactic complexity. Adding two premise vectors increases the overall vector magnitude, producing Euclidean displacement even when the angular direction remains invariant.

This framework suggests that dense sentence embeddings occupy a hyperbolic cone space:
- **Angular Direction**: Represents domain semantics and topic identity.
- **Radial Distance (Magnitude)**: Represents syntactic complexity, clause hierarchy, and sentence scope.

Under this model, Cosine distance acts as an accurate predictor of semantic topic alignment, while Euclidean distance captures the structural complexity required to join multiple ideas into a single compound sentence.

---

## 6. Conclusion & Future Work

In this paper, we investigated the latent geometry of subordinating conjunctions in transformer sentence embeddings. Tracing feature provenance from PERSUADE 2.0 student writing, we showed that subordinating conjunction density is strongly linked to discourse effectiveness and holistic essay quality. Through controlled experiments on a topic-conditioned synthetic dataset of 4,500 dual-premise sentences, we established that:

1. Subordinating conjunction types are non-linearly encoded in dense vector space, enabling deep neural models to achieve 100.00% classification accuracy where linear models fail (57.33%).
2. Dense sentence embeddings exhibit strong additive compositionality: vector addition of constituent premises reconstructs full complex sentences with exceptional cosine accuracy (cosine distance 0.0249).
3. Subordinating connectives function primarily as syntactic operators, introducing minimal angular semantic shift (cosine distance 0.0188) while altering vector magnitude.
4. Sentence embedding space reflects a dual geometry: Cosine distance captures invariant semantic topic orientation, whereas Euclidean distance encodes syntactic magnitude and sentence scope.

These findings confirm that a single dense vector can effectively represent multiple complex ideas simultaneously through linear superimposition. Future work will extend this geometric framework to multi-clause discourse structures, investigating whether hierarchical transformer representations retain additive compositionality across full multi-paragraph essays and complex logical proofs.

---

## References & Data Sources

1. **PERSUADE 2.0 Corpus**: Student persuasive essays and discourse effectiveness annotations. Dataset Payload: `docs/0. Subordinating Conjunction Density (F02) Dataset.md`.
2. **Synthetic Conjunction Dataset**: 4,500 dual-premise sentences across 15 PERSUADE 2.0 topics. Dataset Payload: `docs/1. Synthetic Subordinating Conjunction Embeddings Dataset.md`.
3. **Primary Analysis Notebooks**:
   - `1.subordinating_conjunction_discourse_segmentation.ipynb`
   - `2.subordinating_conjunction_embeddings_dataset_generation.ipynb`
   - `3.subordinating_conjunction_synthetic_dataset_generation.ipynb`
   - `4.subordinating_conjunction_synthetic_embeddings_dataset_generation.ipynb`
   - `5.subordinating_conjunction_classification_and_interpretability.ipynb`
   - `6.subordinating_conjunction_shap_and_attention_interpretability.ipynb`
   - `7.subordinating_conjunction_premise_distance_analysis.ipynb`

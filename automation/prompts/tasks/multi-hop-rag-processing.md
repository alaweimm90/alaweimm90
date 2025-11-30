# Multi-Hop RAG Query Processing

## Purpose

Complex retrieval with reasoning chains for questions requiring multiple evidence pieces.

## Source

RAG research 2024

---

## Multi-Hop Retrieval Process

### 1. QUERY DECOMPOSITION

Break complex query into sub-questions:

```yaml
original_query: "What was the revenue impact of the product launch in Q3 compared to the previous year's Q3?"

decomposed:
  - sub_query_1: 'What was the Q3 revenue this year?'
  - sub_query_2: 'What was the Q3 revenue last year?'
  - sub_query_3: 'What product was launched in Q3?'
  - sub_query_4: "What was the product's contribution to Q3 revenue?"

dependencies:
  - sub_query_4 depends on [sub_query_1, sub_query_3]
  - final_answer depends on [sub_query_1, sub_query_2, sub_query_4]
```

### 2. ITERATIVE RETRIEVAL

Execute retrieval in dependency order:

```python
def multi_hop_retrieve(query, max_hops=5):
    sub_queries = decompose_query(query)
    context = {}

    for hop in range(max_hops):
        # Get next sub-query based on dependencies
        next_query = get_next_executable(sub_queries, context)
        if not next_query:
            break

        # Retrieve relevant documents
        docs = retrieve(next_query, context)

        # Extract answer for sub-query
        answer = extract_answer(next_query, docs)
        context[next_query.id] = answer

        # Check if we can answer original query
        if can_answer_original(query, context):
            break

    return context
```

### 3. EVIDENCE SYNTHESIS

Combine retrieved information:

```yaml
synthesis_steps:
  1. aggregate:
    - Collect all retrieved evidence
    - Tag with source and confidence

  2. resolve_contradictions:
    - Identify conflicting information
    - Prefer more recent sources
    - Prefer authoritative sources
    - Flag unresolved conflicts

  3. check_completeness:
    - Verify all sub-queries answered
    - Identify gaps in evidence
    - Determine if additional retrieval needed

  4. build_reasoning_chain:
    - Connect evidence pieces logically
    - Show derivation of final answer
```

### 4. ANSWER GENERATION

Generate grounded response:

```yaml
generation:
  grounding:
    - Every claim must cite source
    - Use [1], [2] notation for citations
    - Include confidence scores

  structure:
    - Direct answer first
    - Supporting evidence second
    - Caveats and limitations third

  quality_checks:
    - No unsupported claims
    - Logical consistency
    - Addresses original query
```

---

## Self-RAG (Adaptive Retrieval)

### Decision Points

```yaml
retrieval_decision:
  assess: 'Does this query need retrieval?'
  criteria:
    - Factual question → RETRIEVE
    - Opinion/creative → NO RETRIEVE
    - Recent events → RETRIEVE
    - Common knowledge → NO RETRIEVE
    - Domain-specific → RETRIEVE

critique_decision:
  assess: 'Is retrieved content relevant?'
  criteria:
    - Relevance score > 0.7 → USE
    - Relevance score 0.4-0.7 → PARTIAL USE
    - Relevance score < 0.4 → DISCARD + RE-RETRIEVE

generation_decision:
  assess: 'Is generated answer supported?'
  criteria:
    - Fully supported → ACCEPT
    - Partially supported → FLAG UNCERTAINTY
    - Unsupported → REGENERATE or ABSTAIN
```

### Implementation

```python
class SelfRAG:
    def __init__(self, retriever, generator, critic):
        self.retriever = retriever
        self.generator = generator
        self.critic = critic

    def answer(self, query):
        # Decide if retrieval needed
        if self.needs_retrieval(query):
            docs = self.retriever.retrieve(query)

            # Critique retrieval quality
            relevant_docs = self.critic.filter_relevant(docs, query)

            if not relevant_docs:
                # Re-retrieve with reformulated query
                reformulated = self.reformulate_query(query)
                docs = self.retriever.retrieve(reformulated)
                relevant_docs = self.critic.filter_relevant(docs, query)
        else:
            relevant_docs = []

        # Generate answer
        answer = self.generator.generate(query, relevant_docs)

        # Verify answer is supported
        support_score = self.critic.check_support(answer, relevant_docs)

        if support_score < 0.5:
            return self.abstain_or_regenerate(query, relevant_docs)

        return answer, support_score, relevant_docs
```

---

## Agentic RAG Pipeline

### Query Planning

```yaml
query_planner:
  input: complex_user_query

  steps:
    1. intent_classification:
      - factual_lookup
      - comparison
      - aggregation
      - temporal_analysis
      - multi_entity

    2. entity_extraction:
      - named_entities
      - temporal_references
      - numerical_constraints

    3. strategy_selection:
      - single_retrieval
      - iterative_retrieval
      - parallel_retrieval
      - hybrid_retrieval

    4. query_generation:
      - generate_sub_queries
      - optimize_for_retrieval
      - add_filters_and_constraints
```

### Parallel Execution

```python
async def parallel_rag(query, sub_queries):
    # Execute independent sub-queries in parallel
    tasks = [
        retrieve_and_extract(sq)
        for sq in sub_queries
        if not sq.has_dependencies
    ]

    initial_results = await asyncio.gather(*tasks)

    # Execute dependent queries sequentially
    context = dict(zip(
        [sq.id for sq in sub_queries if not sq.has_dependencies],
        initial_results
    ))

    for sq in get_dependent_queries(sub_queries):
        result = await retrieve_and_extract(sq, context)
        context[sq.id] = result

    return context
```

### Verification Layer

```yaml
verification:
  fact_checking:
    - Cross-reference multiple sources
    - Check for temporal validity
    - Verify numerical accuracy

  contradiction_detection:
    - Compare claims across sources
    - Flag inconsistencies
    - Provide confidence intervals

  coverage_analysis:
    - Measure query coverage
    - Identify unanswered aspects
    - Suggest follow-up queries
```

---

## Output Format

```markdown
## Answer

[Direct answer to the query]

## Evidence Chain

### Step 1: [Sub-question 1]

**Retrieved**: [Source document excerpt]
**Finding**: [Extracted answer]
**Confidence**: 0.92

### Step 2: [Sub-question 2]

**Retrieved**: [Source document excerpt]
**Finding**: [Extracted answer]
**Confidence**: 0.87

### Synthesis

[How findings combine to answer original query]

## Sources

1. [Document title, date, relevance score]
2. [Document title, date, relevance score]

## Confidence Assessment

- **Overall Confidence**: 0.85
- **Potential Gaps**: [Any unanswered aspects]
- **Caveats**: [Limitations of the answer]
```

---

## Integration

```bash
# Run multi-hop query
automation rag query "complex question here" --max-hops 5

# With specific sources
automation rag query "question" --sources docs,wiki,api

# Generate report
automation rag report --query "question" --format detailed
```

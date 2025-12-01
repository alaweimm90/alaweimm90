---
name: "AI/ML Integration Superprompt"
version: "1.0"
category: "project"
tags: ["ai", "ml", "llm", "integration", "mlops", "inference"]
created: "2024-11-30"
---

# AI/ML Integration Superprompt

## Purpose

Comprehensive framework for integrating AI/ML capabilities into applications, including LLM integration, model deployment, MLOps practices, and responsible AI implementation.

---

## System Prompt

```text
You are an AI/ML Engineer and Integration Architect with expertise in:
- Large Language Model (LLM) integration and prompt engineering
- ML model deployment and serving (TensorFlow, PyTorch, ONNX)
- MLOps pipelines and model lifecycle management
- Vector databases and semantic search
- RAG (Retrieval-Augmented Generation) systems
- Responsible AI and model governance

Your mission is to build AI integrations that:
1. Leverage state-of-the-art AI capabilities effectively
2. Maintain reliability and performance at scale
3. Implement responsible AI practices
4. Enable continuous model improvement
5. Ensure cost-effective AI operations
```

---

## LLM Integration Patterns

### Multi-Provider LLM Client

```typescript
// lib/ai/llm-client.ts
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';

interface LLMConfig {
  provider: 'anthropic' | 'openai' | 'local';
  model: string;
  temperature?: number;
  maxTokens?: number;
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface LLMResponse {
  content: string;
  usage: {
    inputTokens: number;
    outputTokens: number;
  };
  model: string;
  finishReason: string;
}

export class LLMClient {
  private anthropic?: Anthropic;
  private openai?: OpenAI;
  private config: LLMConfig;

  constructor(config: LLMConfig) {
    this.config = config;
    
    if (config.provider === 'anthropic') {
      this.anthropic = new Anthropic({
        apiKey: process.env.ANTHROPIC_API_KEY,
      });
    } else if (config.provider === 'openai') {
      this.openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
      });
    }
  }

  async chat(messages: Message[], systemPrompt?: string): Promise<LLMResponse> {
    if (this.config.provider === 'anthropic') {
      return this.chatAnthropic(messages, systemPrompt);
    } else if (this.config.provider === 'openai') {
      return this.chatOpenAI(messages, systemPrompt);
    }
    throw new Error(`Unsupported provider: ${this.config.provider}`);
  }

  private async chatAnthropic(
    messages: Message[],
    systemPrompt?: string
  ): Promise<LLMResponse> {
    const response = await this.anthropic!.messages.create({
      model: this.config.model,
      max_tokens: this.config.maxTokens || 4096,
      temperature: this.config.temperature || 0.7,
      system: systemPrompt,
      messages: messages.map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })),
    });

    return {
      content: response.content[0].type === 'text' 
        ? response.content[0].text 
        : '',
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      },
      model: response.model,
      finishReason: response.stop_reason || 'unknown',
    };
  }

  private async chatOpenAI(
    messages: Message[],
    systemPrompt?: string
  ): Promise<LLMResponse> {
    const allMessages = systemPrompt
      ? [{ role: 'system' as const, content: systemPrompt }, ...messages]
      : messages;

    const response = await this.openai!.chat.completions.create({
      model: this.config.model,
      max_tokens: this.config.maxTokens || 4096,
      temperature: this.config.temperature || 0.7,
      messages: allMessages,
    });

    return {
      content: response.choices[0].message.content || '',
      usage: {
        inputTokens: response.usage?.prompt_tokens || 0,
        outputTokens: response.usage?.completion_tokens || 0,
      },
      model: response.model,
      finishReason: response.choices[0].finish_reason || 'unknown',
    };
  }

  // Streaming support
  async *chatStream(
    messages: Message[],
    systemPrompt?: string
  ): AsyncGenerator<string> {
    if (this.config.provider === 'anthropic') {
      const stream = await this.anthropic!.messages.stream({
        model: this.config.model,
        max_tokens: this.config.maxTokens || 4096,
        system: systemPrompt,
        messages: messages.map(m => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
      });

      for await (const event of stream) {
        if (
          event.type === 'content_block_delta' &&
          event.delta.type === 'text_delta'
        ) {
          yield event.delta.text;
        }
      }
    }
  }
}

// Factory function with retry logic
export function createLLMClient(config: LLMConfig): LLMClient {
  return new LLMClient(config);
}
```

### RAG Implementation

```typescript
// lib/ai/rag.ts
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';

interface Document {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
}

interface SearchResult {
  document: Document;
  score: number;
}

export class RAGSystem {
  private pinecone: Pinecone;
  private embeddings: OpenAIEmbeddings;
  private indexName: string;

  constructor(indexName: string) {
    this.pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: 'text-embedding-3-small',
    });
    this.indexName = indexName;
  }

  // Index documents
  async indexDocuments(documents: Document[]): Promise<void> {
    const index = this.pinecone.Index(this.indexName);
    
    // Generate embeddings in batches
    const batchSize = 100;
    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      const texts = batch.map(d => d.content);
      const embeddings = await this.embeddings.embedDocuments(texts);
      
      const vectors = batch.map((doc, idx) => ({
        id: doc.id,
        values: embeddings[idx],
        metadata: {
          content: doc.content,
          ...doc.metadata,
        },
      }));
      
      await index.upsert(vectors);
    }
  }

  // Semantic search
  async search(query: string, topK: number = 5): Promise<SearchResult[]> {
    const index = this.pinecone.Index(this.indexName);
    const queryEmbedding = await this.embeddings.embedQuery(query);
    
    const results = await index.query({
      vector: queryEmbedding,
      topK,
      includeMetadata: true,
    });
    
    return results.matches?.map(match => ({
      document: {
        id: match.id,
        content: match.metadata?.content as string,
        metadata: match.metadata as Record<string, unknown>,
      },
      score: match.score || 0,
    })) || [];
  }

  // RAG query with context
  async query(
    question: string,
    llmClient: LLMClient,
    options: { topK?: number; systemPrompt?: string } = {}
  ): Promise<string> {
    const { topK = 5, systemPrompt } = options;
    
    // Retrieve relevant documents
    const results = await this.search(question, topK);
    
    // Build context
    const context = results
      .map((r, i) => `[${i + 1}] ${r.document.content}`)
      .join('\n\n');
    
    // Generate response with context
    const augmentedPrompt = `
Use the following context to answer the question. If the context doesn't contain 
relevant information, say so and provide a general answer.

Context:
${context}

Question: ${question}

Answer:`;

    const response = await llmClient.chat(
      [{ role: 'user', content: augmentedPrompt }],
      systemPrompt || 'You are a helpful assistant that answers questions based on the provided context.'
    );
    
    return response.content;
  }
}
```

---

## Prompt Engineering

### Prompt Templates

```typescript
// lib/ai/prompts.ts
export const promptTemplates = {
  // Chain of Thought
  chainOfThought: (task: string) => `
Let's approach this step-by-step:

Task: ${task}

Please think through this carefully:
1. First, identify the key components of the problem
2. Then, analyze each component
3. Finally, synthesize your findings into a solution

Show your reasoning at each step.`,

  // Few-Shot Learning
  fewShot: (examples: Array<{ input: string; output: string }>, newInput: string) => `
Here are some examples:

${examples.map((ex, i) => `Example ${i + 1}:
Input: ${ex.input}
Output: ${ex.output}`).join('\n\n')}

Now, please process this new input:
Input: ${newInput}
Output:`,

  // Self-Consistency
  selfConsistency: (question: string, numPaths: number = 3) => `
Please solve this problem using ${numPaths} different approaches, then determine 
the most likely correct answer based on consistency across approaches.

Question: ${question}

Approach 1:
[Solve using first method]

Approach 2:
[Solve using second method]

Approach 3:
[Solve using third method]

Final Answer (based on consistency):`,

  // ReAct Pattern
  react: (task: string, tools: string[]) => `
You have access to the following tools: ${tools.join(', ')}

Use the following format:

Thought: Consider what to do next
Action: The action to take (one of: ${tools.join(', ')})
Action Input: The input to the action
Observation: The result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: The final answer to the task

Task: ${task}

Begin!`,

  // Structured Output
  structuredOutput: (task: string, schema: object) => `
${task}

Please provide your response in the following JSON format:
${JSON.stringify(schema, null, 2)}

Ensure your response is valid JSON that matches this schema exactly.`,

  // Code Generation
  codeGeneration: (
    language: string,
    task: string,
    context?: string
  ) => `
Generate ${language} code for the following task:

${task}

${context ? `Context:\n${context}\n` : ''}

Requirements:
- Write clean, well-documented code
- Include error handling
- Follow ${language} best practices
- Add type annotations where applicable

Code:`,
};

// Prompt builder with validation
export class PromptBuilder {
  private parts: string[] = [];
  private variables: Map<string, string> = new Map();

  system(content: string): this {
    this.parts.push(`<system>\n${content}\n</system>`);
    return this;
  }

  context(content: string): this {
    this.parts.push(`<context>\n${content}\n</context>`);
    return this;
  }

  examples(examples: Array<{ input: string; output: string }>): this {
    const examplesText = examples
      .map((ex, i) => `Example ${i + 1}:\nInput: ${ex.input}\nOutput: ${ex.output}`)
      .join('\n\n');
    this.parts.push(`<examples>\n${examplesText}\n</examples>`);
    return this;
  }

  task(content: string): this {
    this.parts.push(`<task>\n${content}\n</task>`);
    return this;
  }

  constraints(items: string[]): this {
    const constraintsText = items.map(c => `- ${c}`).join('\n');
    this.parts.push(`<constraints>\n${constraintsText}\n</constraints>`);
    return this;
  }

  outputFormat(format: string | object): this {
    const formatText = typeof format === 'string' 
      ? format 
      : JSON.stringify(format, null, 2);
    this.parts.push(`<output_format>\n${formatText}\n</output_format>`);
    return this;
  }

  variable(name: string, value: string): this {
    this.variables.set(name, value);
    return this;
  }

  build(): string {
    let prompt = this.parts.join('\n\n');
    
    // Replace variables
    for (const [name, value] of this.variables) {
      prompt = prompt.replace(new RegExp(`{{${name}}}`, 'g'), value);
    }
    
    return prompt;
  }
}
```

---

## Model Serving

### FastAPI Model Server

```python
# model_server/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
import time

app = FastAPI(title="ML Model Server")

# Metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_name', 'status']
)
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name']
)

# Model registry
models = {}
tokenizers = {}

class PredictionRequest(BaseModel):
    text: str
    model_name: str = "default"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    latency_ms: float

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    model_name: str = "default"

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    model_configs = [
        {"name": "default", "path": "distilbert-base-uncased-finetuned-sst-2-english"},
    ]
    
    for config in model_configs:
        models[config["name"]] = AutoModelForSequenceClassification.from_pretrained(
            config["path"]
        )
        tokenizers[config["name"]] = AutoTokenizer.from_pretrained(config["path"])
        models[config["name"]].eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            models[config["name"]].cuda()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    start_time = time.time()
    
    if request.model_name not in models:
        PREDICTION_COUNTER.labels(model_name=request.model_name, status="error").inc()
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    try:
        model = models[request.model_name]
        tokenizer = tokenizers[request.model_name]
        
        # Tokenize
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        latency_ms = (time.time() - start_time) * 1000
        
        PREDICTION_COUNTER.labels(model_name=request.model_name, status="success").inc()
        PREDICTION_LATENCY.labels(model_name=request.model_name).observe(latency_ms / 1000)
        
        return PredictionResponse(
            prediction=model.config.id2label[prediction],
            confidence=confidence,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        PREDICTION_COUNTER.labels(model_name=request.model_name, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    model = models[request.model_name]
    tokenizer = tokenizers[request.model_name]
    
    # Tokenize batch
    inputs = tokenizer(
        request.texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()
    
    return {
        "predictions": [
            {
                "text": text,
                "prediction": model.config.id2label[pred],
                "confidence": conf
            }
            for text, pred, conf in zip(request.texts, predictions, confidences)
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()
```

---

## MLOps Pipeline

### Model Training Pipeline

```yaml
# mlops/training-pipeline.yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: model-training-pipeline
spec:
  entrypoint: training-pipeline
  
  templates:
    - name: training-pipeline
      dag:
        tasks:
          - name: data-validation
            template: validate-data
            
          - name: feature-engineering
            template: engineer-features
            dependencies: [data-validation]
            
          - name: train-model
            template: train
            dependencies: [feature-engineering]
            
          - name: evaluate-model
            template: evaluate
            dependencies: [train-model]
            
          - name: register-model
            template: register
            dependencies: [evaluate-model]
            when: "{{tasks.evaluate-model.outputs.parameters.passed}} == true"
            
          - name: deploy-model
            template: deploy
            dependencies: [register-model]

    - name: validate-data
      container:
        image: ml-pipeline:latest
        command: [python, -m, pipeline.validate_data]
        args:
          - --input-path={{workflow.parameters.data-path}}
          - --output-path=/tmp/validated-data
          
    - name: engineer-features
      container:
        image: ml-pipeline:latest
        command: [python, -m, pipeline.feature_engineering]
        args:
          - --input-path=/tmp/validated-data
          - --output-path=/tmp/features
          
    - name: train
      container:
        image: ml-pipeline:latest
        command: [python, -m, pipeline.train]
        args:
          - --features-path=/tmp/features
          - --model-path=/tmp/model
          - --experiment-name={{workflow.parameters.experiment-name}}
        resources:
          limits:
            nvidia.com/gpu: 1
            
    - name: evaluate
      container:
        image: ml-pipeline:latest
        command: [python, -m, pipeline.evaluate]
        args:
          - --model-path=/tmp/model
          - --test-data=/tmp/features/test
      outputs:
        parameters:
          - name: passed
            valueFrom:
              path: /tmp/evaluation-passed
              
    - name: register
      container:
        image: ml-pipeline:latest
        command: [python, -m, pipeline.register_model]
        args:
          - --model-path=/tmp/model
          - --model-name={{workflow.parameters.model-name}}
          
    - name: deploy
      container:
        image: ml-pipeline:latest
        command: [python, -m, pipeline.deploy]
        args:
          - --model-name={{workflow.parameters.model-name}}
          - --environment=staging
```

### Model Monitoring

```python
# monitoring/model_monitor.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats
from datetime import datetime
import json

@dataclass
class DriftReport:
    feature_name: str
    drift_score: float
    drift_detected: bool
    p_value: float
    reference_stats: Dict
    current_stats: Dict
    timestamp: datetime

class ModelMonitor:
    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: List[str],
        drift_threshold: float = 0.05
    ):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.drift_threshold = drift_threshold
        self.reference_stats = self._compute_stats(reference_data)
        
    def _compute_stats(self, data: np.ndarray) -> Dict:
        """Compute statistical summary of data."""
        return {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'percentiles': {
                '25': np.percentile(data, 25, axis=0).tolist(),
                '50': np.percentile(data, 50, axis=0).tolist(),
                '75': np.percentile(data, 75, axis=0).tolist(),
            }
        }
    
    def detect_drift(self, current_data: np.ndarray) -> List[DriftReport]:
        """Detect data drift using Kolmogorov-Smirnov test."""
        reports = []
        current_stats = self._compute_stats(current_data)
        
        for i, feature_name in enumerate(self.feature_names):
            # KS test for each feature
            statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                current_data[:, i]
            )
            
            drift_detected = p_value < self.drift_threshold
            
            reports.append(DriftReport(
                feature_name=feature_name,
                drift_score=statistic,
                drift_detected=drift_detected,
                p_value=p_value,
                reference_stats={
                    'mean': self.reference_stats['mean'][i],
                    'std': self.reference_stats['std'][i],
                },
                current_stats={
                    'mean': current_stats['mean'][i],
                    'std': current_stats['std'][i],
                },
                timestamp=datetime.utcnow()
            ))
            
        return reports
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict:
        """Detect drift in model predictions."""
        statistic, p_value = stats.ks_2samp(
            reference_predictions,
            current_predictions
        )
        
        return {
            'drift_score': statistic,
            'p_value': p_value,
            'drift_detected': p_value < self.drift_threshold,
            'reference_distribution': {
                'mean': float(np.mean(reference_predictions)),
                'std': float(np.std(reference_predictions)),
            },
            'current_distribution': {
                'mean': float(np.mean(current_predictions)),
                'std': float(np.std(current_predictions)),
            }
        }
    
    def check_data_quality(self, data: np.ndarray) -> Dict:
        """Check data quality metrics."""
        return {
            'missing_values': int(np.isnan(data).sum()),
            'missing_percentage': float(np.isnan(data).mean() * 100),
            'infinite_values': int(np.isinf(data).sum()),
            'out_of_range': self._check_out_of_range(data),
            'duplicate_rows': self._count_duplicates(data),
        }
    
    def _check_out_of_range(self, data: np.ndarray) -> Dict:
        """Check for values outside reference range."""
        out_of_range = {}
        for i, feature_name in enumerate(self.feature_names):
            ref_min = self.reference_stats['min'][i]
            ref_max = self.reference_stats['max'][i]
            
            below_min = (data[:, i] < ref_min).sum()
            above_max = (data[:, i] > ref_max).sum()
            
            if below_min > 0 or above_max > 0:
                out_of_range[feature_name] = {
                    'below_min': int(below_min),
                    'above_max': int(above_max),
                }
                
        return out_of_range
    
    def _count_duplicates(self, data: np.ndarray) -> int:
        """Count duplicate rows."""
        unique_rows = np.unique(data, axis=0)
        return len(data) - len(unique_rows)
```

---

## Execution Phases

### Phase 1: LLM Integration

- [ ] Set up LLM client with multiple providers
- [ ] Implement prompt templates
- [ ] Add streaming support
- [ ] Configure rate limiting and caching

### Phase 2: RAG System

- [ ] Set up vector database
- [ ] Implement document indexing
- [ ] Build semantic search
- [ ] Create RAG query pipeline

### Phase 3: Model Serving

- [ ] Deploy model server
- [ ] Add batch prediction
- [ ] Configure auto-scaling
- [ ] Set up monitoring

### Phase 4: MLOps

- [ ] Build training pipeline
- [ ] Implement model registry
- [ ] Add drift detection
- [ ] Configure alerting

---

*Last updated: 2024-11-30*

# Composeable NLP Components Architecture

## Overview

This project now features a **composeable architecture** where each NLP component can be:
- **Run independently** with its own CLI
- **Composed together** in pipelines
- **Tested in isolation**
- **Configured independently**

This architecture promotes:
- **Modularity**: Each component is self-contained
- **Reusability**: Components can be reused across different systems
- **Testability**: Components can be tested independently
- **Flexibility**: Easy to swap or upgrade individual components

---

## Component Hierarchy

```
src/components/
├── base.py                      # Base interfaces (Component, ComponentInput, ComponentOutput)
├── ner/                         # Named Entity Recognition
│   ├── __init__.py
│   └── component.py             # Standalone NER component
├── classification/              # Intent Classification
│   ├── __init__.py
│   └── component.py             # Standalone classification component
├── semantic_search/             # Semantic Search (FAISS)
│   ├── __init__.py
│   └── component.py             # Standalone search component
├── llm/                         # Language Model (Abstractive)
│   ├── __init__.py
│   └── component.py             # Low-cost LLM component
└── dialogue/                    # Dialogue Manager
    ├── __init__.py
    └── manager.py               # Central orchestrator
```

---

## Base Component Interface

All components implement the `Component` protocol:

```python
from src.components.base import Component, ComponentInput, ComponentOutput, ComponentConfig

class Component(ABC):
    async def initialize(self) -> None:
        """Initialize the component (load models, etc.)"""
        pass

    async def process(self, input_data: ComponentInput) -> ComponentOutput:
        """Process input and return output"""
        pass

    async def process_batch(self, inputs: list[ComponentInput]) -> list[ComponentOutput]:
        """Process multiple inputs in batch"""
        pass

    def get_name(self) -> str:
        """Return component name"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the component"""
        pass
```

---

## Components

### 1. NER Component

**Purpose**: Extract named entities from text using hybrid approach (transformers + regex).

**Features**:
- Transformer-based NER (XLM-RoBERTa)
- Regex patterns for structured data (NID, phone, email, etc.)
- Entity-to-slot mapping for forms
- Can run in regex-only mode (lightweight)

**Usage**:
```python
from src.components.ner import NERComponent, NERInput, NERConfig

config = NERConfig(device="cpu", enable_transformer=True)
ner = NERComponent(config)
await ner.initialize()

result = await ner.process(NERInput(
    text="আমার নাম জন ডো এবং আমার NID 12345678901234567"
))

print(result.entities)  # {"person_name": "জন ডো", "nid_number": "12345678901234567"}
```

**CLI**:
```bash
python cli.py ner --text "আমার নাম জন ডো" --device cpu
```

---

### 2. Classification Component

**Purpose**: Classify user queries into intent clusters using E5 embeddings + Logistic Regression.

**Features**:
- E5 multilingual embeddings
- 48-class intent classification
- Top-K predictions with confidence scores
- Async processing with thread pool

**Usage**:
```python
from src.components.classification import ClassificationComponent, ClassificationInput, ClassificationConfig

config = ClassificationConfig(device="cpu")
classifier = ClassificationComponent(config)
await classifier.initialize()

result = await classifier.process(ClassificationInput(
    text="আমার NID স্ট্যাটাস কি?"
))

print(result.top_cluster, result.top_confidence)
```

**CLI**:
```bash
python cli.py classification --text "আমার NID স্ট্যাটাস কি?" --top-k 3
```

---

### 3. Semantic Search Component

**Purpose**: Search for similar questions using FAISS indices and E5 embeddings.

**Features**:
- FAISS-based similarity search
- Cluster-specific or merged search
- Top-K results with scores
- Pre-built indices for 3.2k+ Bengali FAQs

**Usage**:
```python
from src.components.semantic_search import SemanticSearchComponent, SemanticSearchInput, SemanticSearchConfig

config = SemanticSearchConfig(device="cpu")
searcher = SemanticSearchComponent(config)
await searcher.initialize()

result = await searcher.process(SemanticSearchInput(
    text="NID কিভাবে আবেদন করবো?",
    cluster="nid_application"  # Optional
))

print(result.results)  # List of similar questions
```

**CLI**:
```bash
python cli.py search --text "NID কিভাবে আবেদন করবো?" --top-k 5
```

---

### 4. LLM Component

**Purpose**: Low-cost abstractive text generation for summarization, query expansion, response generation.

**Features**:
- Multiple backends: Local (Flan-T5, Phi-3), OpenAI, Anthropic
- Response caching to reduce costs
- Task-specific prompting
- Configurable max tokens and temperature

**Usage**:
```python
from src.components.llm import LLMComponent, LLMInput, LLMConfig, LLMBackend

# Local model (free, slower)
config = LLMConfig(
    backend=LLMBackend.LOCAL,
    model_name="google/flan-t5-base",
    device="cpu"
)

# Or OpenAI (fast, paid)
config = LLMConfig(
    backend=LLMBackend.OPENAI,
    model_name="gpt-3.5-turbo",
    api_key="your-key"
)

llm = LLMComponent(config)
await llm.initialize()

result = await llm.process(LLMInput(
    text="Summarize this: ...",
    task_type="summarization"
))

print(result.generated_text)
```

**CLI**:
```bash
# Local model
python cli.py llm --text "Summarize this text" --backend local

# OpenAI
python cli.py llm --text "Summarize this text" --backend openai --api-key YOUR_KEY
```

---

### 5. Dialogue Manager

**Purpose**: Central orchestrator for multi-turn conversations integrating all NLP components.

**Features**:
- Multi-turn conversation management
- Text fragmentation for long inputs
- Context-aware processing (fractional query detection)
- Session state tracking
- Action invocation framework
- Composeable with all NLP components

**Usage**:
```python
from src.components.dialogue import DialogueManager, DialogueInput, DialogueConfig

config = DialogueConfig(
    device="cpu",
    enable_ner=True,
    enable_classification=True,
    enable_search=True,
    enable_llm=False
)

dm = DialogueManager(config)
await dm.initialize()

# First turn
result = await dm.process(DialogueInput(
    text="আমার NID নাম্বার 12345678901234567",
    session_id="user123"
))

# Second turn (fractional query)
result = await dm.process(DialogueInput(
    text="আমার স্ট্যাটাস কি?",
    session_id="user123"  # Same session
))

print(result.response, result.entities, result.intent)
```

**CLI**:
```bash
# Interactive mode
python cli.py dialogue --interactive --device cpu

# Single query
python cli.py dialogue --text "আমার NID স্ট্যাটাস কি?" --session-id user123
```

---

## Component Composition

### Pipeline Pattern

Compose components in a sequential pipeline:

```python
from src.components.base import ComponentPipeline, ComponentInput

# Create components
ner = NERComponent(NERConfig())
classifier = ClassificationComponent(ClassificationConfig())
searcher = SemanticSearchComponent(SemanticSearchConfig())

# Create pipeline
pipeline = ComponentPipeline([ner, classifier, searcher])
await pipeline.initialize_all()

# Process through pipeline
outputs = await pipeline.process(ComponentInput(text="..."))

# Each component's output is available in context for next component
```

### Custom Composition

Create custom workflows:

```python
# 1. Classify intent
cls_result = await classifier.process(ClassificationInput(text=query))

# 2. Search in predicted cluster
search_result = await searcher.process(SemanticSearchInput(
    text=query,
    cluster=cls_result.top_cluster
))

# 3. Extract entities if needed
if needs_entities:
    ner_result = await ner.process(NERInput(text=query))

# 4. Generate response with LLM
llm_result = await llm.process(LLMInput(
    prompt=f"Answer based on: {search_result.results[0]}",
))
```

---

## Configuration

Each component has its own config class:

```python
# NER Config
NERConfig(
    model_name="xlm-roberta-large-finetuned-conll03-english",
    confidence_threshold=0.7,
    enable_regex_fallback=True,
    enable_transformer=True,
    device="cpu"
)

# Classification Config
ClassificationConfig(
    model_path="models/classification/model.pkl",
    label_encoder_path="models/classification/label_encoder.pkl",
    embedding_model="intfloat/multilingual-e5-large-instruct",
    top_k=3,
    confidence_threshold=0.6,
    device="cpu"
)

# Search Config
SemanticSearchConfig(
    indices_dir="models/semantic_search/faiss_indices",
    metadata_path="models/semantic_search/cluster_metadata.json",
    embedding_model="intfloat/multilingual-e5-large-instruct",
    top_k=5,
    device="cpu"
)

# LLM Config
LLMConfig(
    backend=LLMBackend.LOCAL,  # or OPENAI, ANTHROPIC
    model_name="google/flan-t5-base",
    max_tokens=256,
    temperature=0.7,
    cache_responses=True,
    device="cpu"
)

# Dialogue Config
DialogueConfig(
    enable_ner=True,
    enable_classification=True,
    enable_search=True,
    enable_llm=False,
    enable_text_fragmentation=True,
    max_fragment_length=512,
    enable_context_augmentation=True,
    max_context_turns=3,
    device="cpu"
)
```

---

## Input/Output Formats

All components use standardized JSON-serializable formats:

### Input
```python
@dataclass
class ComponentInput:
    text: str
    language: str = "bn"
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    tenant_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, json_str: str) -> "ComponentInput": ...
```

### Output
```python
@dataclass
class ComponentOutput:
    success: bool
    data: Any
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    component_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, json_str: str) -> "ComponentOutput": ...
```

---

## Testing Components

Each component can be tested independently:

```python
import pytest
from src.components.ner import NERComponent, NERInput, NERConfig

@pytest.mark.asyncio
async def test_ner_component():
    config = NERConfig(enable_transformer=False)  # Fast regex-only mode
    ner = NERComponent(config)
    await ner.initialize()

    result = await ner.process(NERInput(
        text="আমার NID 12345678901234567"
    ))

    assert result.success
    assert "nid_number" in result.entities
    assert result.entities["nid_number"] == "12345678901234567"
```

---

## Health Checks

All components support health checks:

```python
health = await component.health_check()

# Returns:
{
    "name": "ner",
    "initialized": True,
    "config": {...},
    "status": "healthy",
    # Component-specific fields...
}
```

---

## Example: Complete Demo

See `examples/composeable_components_demo.py` for a complete demonstration of:
1. Running components independently
2. Composing components in pipelines
3. Using the Dialogue Manager for multi-turn conversations

Run it with:
```bash
python examples/composeable_components_demo.py
```

---

## Migration from Old Architecture

The old monolithic `EnhancedDialoguePipeline` is still available in `src/application/dialogue_service.py`.

To migrate to the new architecture:

1. **Replace monolithic pipeline** with Dialogue Manager
2. **Use individual components** for specific tasks
3. **Compose as needed** for your use case

Benefits:
- Better testability
- Easier to maintain
- More flexible
- Can run components independently
- Can swap implementations easily

---

## Performance Considerations

### Lazy Loading
- Components load models only when `initialize()` is called
- Can initialize selectively (e.g., only NER for extraction tasks)

### Async Processing
- All components support async processing
- CPU-bound operations run in thread pools
- Non-blocking I/O

### Batch Processing
- All components support batch processing via `process_batch()`
- Override for better performance in your use case

### Caching
- LLM component caches responses to reduce costs
- E5 embeddings cached by sentence-transformers
- FAISS indices loaded once and reused

---

## Future Enhancements

1. **More Backends**: Add support for more LLM backends (Cohere, Together.ai, etc.)
2. **Streaming**: Add streaming support for LLM component
3. **Metrics**: Add built-in metrics collection for each component
4. **Persistence**: Add session persistence (Redis, PostgreSQL)
5. **Distributed**: Add support for running components as microservices
6. **More Components**: Add more NLP components (sentiment, translation, etc.)

---

## Contributing

When adding new components:

1. Inherit from `Component` base class
2. Implement required methods (`initialize`, `process`, `get_name`)
3. Use typed Input/Output dataclasses
4. Add CLI entry point
5. Add health check
6. Add tests
7. Document in this file

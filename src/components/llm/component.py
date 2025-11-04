"""
Standalone LLM Component implementation with multiple backends.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from src.components.base import Component, ComponentInput, ComponentOutput, ComponentConfig

logger = logging.getLogger(__name__)


class LLMBackend(str, Enum):
    """Supported LLM backends"""
    LOCAL = "local"  # Local transformers model (Flan-T5, Phi-3, etc.)
    OPENAI = "openai"  # OpenAI API
    ANTHROPIC = "anthropic"  # Anthropic API


@dataclass
class LLMConfig(ComponentConfig):
    """Configuration for LLM component"""
    backend: LLMBackend = LLMBackend.LOCAL
    model_name: str = "google/flan-t5-base"  # For LOCAL backend
    api_key: Optional[str] = None  # For API backends
    max_tokens: int = 256
    temperature: float = 0.7
    cache_responses: bool = True  # Cache responses to save costs
    cache_ttl_seconds: int = 3600  # Cache TTL
    system_prompt: Optional[str] = None  # System prompt for chat models


@dataclass
class LLMInput(ComponentInput):
    """Input for LLM component"""
    prompt: str = ""  # Main prompt/instruction
    max_tokens: Optional[int] = None  # Override config max_tokens
    temperature: Optional[float] = None  # Override config temperature
    task_type: str = "generation"  # "summarization", "expansion", "generation", etc.

    def __post_init__(self):
        # If prompt not provided, use text as prompt
        if not self.prompt and self.text:
            self.prompt = self.text


@dataclass
class LLMOutput(ComponentOutput):
    """Output from LLM component"""
    generated_text: str = ""
    tokens_used: int = 0
    cached: bool = False
    backend: str = ""

    def __post_init__(self):
        # Ensure data contains generated text
        if self.data is None or not isinstance(self.data, dict):
            self.data = {
                "generated_text": self.generated_text,
                "tokens_used": self.tokens_used,
                "cached": self.cached,
                "backend": self.backend,
            }


class LLMComponent(Component[LLMInput, LLMOutput, LLMConfig]):
    """
    Standalone LLM Component with multiple backends.

    Features:
    - Local model support (Flan-T5, Phi-3, etc.)
    - API support (OpenAI, Anthropic)
    - Response caching to reduce costs
    - Task-specific prompting
    - Can run independently or in pipeline

    Example usage:
        # Local model
        config = LLMConfig(
            backend=LLMBackend.LOCAL,
            model_name="google/flan-t5-base",
            device="cpu"
        )
        llm = LLMComponent(config)
        await llm.initialize()

        input_data = LLMInput(
            text="Summarize this: ...",
            task_type="summarization"
        )
        output = await llm.process(input_data)
        print(output.generated_text)
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.cache: Dict[str, tuple] = {}  # prompt -> (response, timestamp)

    async def initialize(self) -> None:
        """Initialize LLM backend"""
        if self._initialized:
            logger.info("LLM component already initialized")
            return

        logger.info(f"Initializing LLM component with backend: {self.config.backend}")
        start_time = time.time()

        if self.config.backend == LLMBackend.LOCAL:
            await self._initialize_local()
        elif self.config.backend == LLMBackend.OPENAI:
            await self._initialize_openai()
        elif self.config.backend == LLMBackend.ANTHROPIC:
            await self._initialize_anthropic()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        self._initialized = True
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"LLM component initialized in {elapsed:.2f}ms")

    async def _initialize_local(self) -> None:
        """Initialize local transformer model"""
        logger.info(f"Loading local model: {self.config.model_name}")

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
            import torch

            # Try Seq2Seq first (for T5-style models)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.model_path
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.model_path
                )
                logger.info("Loaded as Seq2Seq model")
            except:
                # Fall back to CausalLM (for GPT-style models)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.model_path
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.model_path
                )
                logger.info("Loaded as CausalLM model")

            # Move to device
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("Model moved to CUDA")
            else:
                self.model = self.model.to("cpu")
                logger.info("Model on CPU")

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    async def _initialize_openai(self) -> None:
        """Initialize OpenAI client"""
        if not self.config.api_key:
            raise ValueError("OpenAI API key required")

        logger.info("OpenAI backend ready")
        # OpenAI client initialized on-demand

    async def _initialize_anthropic(self) -> None:
        """Initialize Anthropic client"""
        if not self.config.api_key:
            raise ValueError("Anthropic API key required")

        logger.info("Anthropic backend ready")
        # Anthropic client initialized on-demand

    async def process(self, input_data: LLMInput) -> LLMOutput:
        """Generate text using LLM"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Check cache first
            if self.config.cache_responses:
                cached_response = self._get_cached_response(input_data.prompt)
                if cached_response:
                    processing_time_ms = (time.time() - start_time) * 1000
                    logger.info("Returning cached response")
                    return LLMOutput(
                        success=True,
                        data={
                            "generated_text": cached_response["text"],
                            "tokens_used": cached_response["tokens"],
                            "cached": True,
                            "backend": self.config.backend.value,
                        },
                        generated_text=cached_response["text"],
                        tokens_used=cached_response["tokens"],
                        cached=True,
                        backend=self.config.backend.value,
                        processing_time_ms=processing_time_ms,
                        component_name=self.get_name(),
                    )

            # Generate new response
            if self.config.backend == LLMBackend.LOCAL:
                generated_text, tokens_used = await self._generate_local(input_data)
            elif self.config.backend == LLMBackend.OPENAI:
                generated_text, tokens_used = await self._generate_openai(input_data)
            elif self.config.backend == LLMBackend.ANTHROPIC:
                generated_text, tokens_used = await self._generate_anthropic(input_data)
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")

            # Cache the response
            if self.config.cache_responses:
                self._cache_response(input_data.prompt, generated_text, tokens_used)

            processing_time_ms = (time.time() - start_time) * 1000

            return LLMOutput(
                success=True,
                data={
                    "generated_text": generated_text,
                    "tokens_used": tokens_used,
                    "cached": False,
                    "backend": self.config.backend.value,
                },
                generated_text=generated_text,
                tokens_used=tokens_used,
                cached=False,
                backend=self.config.backend.value,
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

        except Exception as e:
            logger.error(f"LLM processing failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return LLMOutput(
                success=False,
                data={},
                error=str(e),
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

    async def _generate_local(self, input_data: LLMInput) -> tuple[str, int]:
        """Generate text using local model"""
        import torch
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def _generate_sync():
            max_tokens = input_data.max_tokens or self.config.max_tokens
            temperature = input_data.temperature or self.config.temperature

            # Tokenize input
            inputs = self.tokenizer(
                input_data.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            # Move to device
            if self.config.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # For Seq2Seq models, the output is the generated text
            # For CausalLM, we need to remove the input prompt
            if hasattr(self.model, "generate") and not hasattr(self.model, "encoder"):
                # CausalLM - remove input
                input_length = inputs["input_ids"].shape[1]
                output_ids = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            tokens_used = outputs.shape[1]

            return generated_text.strip(), tokens_used

        # Run in thread pool to avoid blocking
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _generate_sync)

        return result

    async def _generate_openai(self, input_data: LLMInput) -> tuple[str, int]:
        """Generate text using OpenAI API"""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for OpenAI backend: pip install openai")

        client = openai.AsyncOpenAI(api_key=self.config.api_key)

        max_tokens = input_data.max_tokens or self.config.max_tokens
        temperature = input_data.temperature or self.config.temperature

        response = await client.chat.completions.create(
            model=self.config.model_name or "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.config.system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": input_data.prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        generated_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return generated_text.strip(), tokens_used

    async def _generate_anthropic(self, input_data: LLMInput) -> tuple[str, int]:
        """Generate text using Anthropic API"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required for Anthropic backend: pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=self.config.api_key)

        max_tokens = input_data.max_tokens or self.config.max_tokens
        temperature = input_data.temperature or self.config.temperature

        response = await client.messages.create(
            model=self.config.model_name or "claude-3-haiku-20240307",
            max_tokens=max_tokens,
            temperature=temperature,
            system=self.config.system_prompt or "You are a helpful assistant.",
            messages=[
                {"role": "user", "content": input_data.prompt}
            ]
        )

        generated_text = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        return generated_text.strip(), tokens_used

    def _get_cached_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        if prompt not in self.cache:
            return None

        cached_data, timestamp = self.cache[prompt]

        # Check if expired
        if time.time() - timestamp > self.config.cache_ttl_seconds:
            del self.cache[prompt]
            return None

        return cached_data

    def _cache_response(self, prompt: str, text: str, tokens: int) -> None:
        """Cache a response"""
        self.cache[prompt] = (
            {"text": text, "tokens": tokens},
            time.time()
        )

    def get_name(self) -> str:
        """Return component name"""
        return "llm"

    async def health_check(self) -> Dict[str, Any]:
        """Health check for LLM component"""
        base_health = await super().health_check()

        llm_specific = {
            "backend": self.config.backend.value,
            "model_name": self.config.model_name,
            "cached_responses": len(self.cache),
        }

        return {**base_health, **llm_specific}


# CLI entry point
async def main():
    """CLI entry point for standalone LLM component"""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Standalone LLM Component")
    parser.add_argument("--text", type=str, help="Text/prompt to process")
    parser.add_argument("--input-file", type=str, help="Input JSON file")
    parser.add_argument("--output-file", type=str, help="Output JSON file")
    parser.add_argument("--backend", type=str, default="local", choices=["local", "openai", "anthropic"])
    parser.add_argument("--model-name", type=str, default="google/flan-t5-base")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI/Anthropic")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--task-type", type=str, default="generation")

    args = parser.parse_args()

    # Create config
    config = LLMConfig(
        backend=LLMBackend(args.backend),
        model_name=args.model_name,
        api_key=args.api_key,
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Initialize component
    component = LLMComponent(config)
    await component.initialize()

    # Prepare input
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_json = json.load(f)
            input_data = LLMInput(**input_json)
    elif args.text:
        input_data = LLMInput(text=args.text, task_type=args.task_type)
    else:
        print("Error: Must provide --text or --input-file", file=sys.stderr)
        sys.exit(1)

    # Process
    output = await component.process(input_data)

    # Write output
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output.to_json())
    else:
        print(output.to_json())

    # Exit with appropriate code
    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

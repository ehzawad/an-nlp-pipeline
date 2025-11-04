"""
Dialogue Manager - Central conversation orchestrator.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import time
import uuid

from src.components.base import Component, ComponentInput, ComponentOutput, ComponentConfig

# Import NLP components
from src.components.ner import NERComponent, NERInput, NERConfig
from src.components.classification import ClassificationComponent, ClassificationInput, ClassificationConfig
from src.components.semantic_search import SemanticSearchComponent, SemanticSearchInput, SemanticSearchConfig
from src.components.llm import LLMComponent, LLMInput, LLMConfig

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message role in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """Single message in conversation"""
    role: MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """Session state for multi-turn conversation"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # Accumulated entities
    context: Dict[str, Any] = field(default_factory=dict)  # Conversation context
    current_intent: Optional[str] = None
    active_form: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict] = None):
        """Add message to history"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow().isoformat()

    def update_entities(self, new_entities: Dict[str, str]):
        """Update accumulated entities"""
        self.entities.update(new_entities)
        self.updated_at = datetime.utcnow().isoformat()

    def get_recent_context(self, max_turns: int = 3) -> str:
        """Get recent conversation context"""
        recent_messages = self.messages[-max_turns * 2:] if len(self.messages) > 0 else []
        context_parts = []
        for msg in recent_messages:
            context_parts.append(f"{msg.role.value}: {msg.content}")
        return "\n".join(context_parts)


@dataclass
class DialogueConfig(ComponentConfig):
    """Configuration for Dialogue Manager"""
    # Component configs
    ner_config: Optional[NERConfig] = None
    classification_config: Optional[ClassificationConfig] = None
    search_config: Optional[SemanticSearchConfig] = None
    llm_config: Optional[LLMConfig] = None

    # Session settings
    session_store: str = "memory"  # "memory", "redis", "file"
    session_ttl_seconds: int = 3600

    # Processing settings
    enable_ner: bool = True
    enable_classification: bool = True
    enable_search: bool = True
    enable_llm: bool = False  # LLM is optional
    enable_text_fragmentation: bool = True
    max_fragment_length: int = 512

    # Context settings
    enable_context_augmentation: bool = True
    max_context_turns: int = 3
    fractional_query_threshold: float = 0.7


@dataclass
class DialogueInput(ComponentInput):
    """Input for Dialogue Manager"""
    # Optional: provide existing session_id to continue conversation
    # session_id is in parent ComponentInput

    # Optional: specific actions to perform
    extract_entities: bool = True
    classify_intent: bool = True
    search_knowledge: bool = True
    generate_response: bool = True


@dataclass
class DialogueOutput(ComponentOutput):
    """Output from Dialogue Manager"""
    response: str = ""
    session_id: str = ""
    entities: Dict[str, str] = field(default_factory=dict)
    intent: Optional[str] = None
    confidence: float = 0.0
    actions: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_results: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Ensure data contains all fields
        if self.data is None or not isinstance(self.data, dict):
            self.data = {
                "response": self.response,
                "session_id": self.session_id,
                "entities": self.entities,
                "intent": self.intent,
                "confidence": self.confidence,
                "actions": self.actions,
                "knowledge_results": self.knowledge_results,
            }


class DialogueManager(Component[DialogueInput, DialogueOutput, DialogueConfig]):
    """
    Dialogue Manager - Central conversation orchestrator.

    Features:
    - Multi-turn conversation management
    - Text fragmentation for long inputs
    - NER, classification, search integration
    - Optional LLM for response generation
    - Session state tracking
    - Action invocation framework

    Example usage:
        config = DialogueConfig(
            enable_ner=True,
            enable_classification=True,
            enable_search=True,
        )
        dm = DialogueManager(config)
        await dm.initialize()

        input_data = DialogueInput(
            text="আমার NID নাম্বার 12345678901234567",
            session_id="user123"
        )
        output = await dm.process(input_data)
        print(output.response, output.entities)
    """

    def __init__(self, config: DialogueConfig):
        super().__init__(config)

        # NLP Components
        self.ner: Optional[NERComponent] = None
        self.classifier: Optional[ClassificationComponent] = None
        self.searcher: Optional[SemanticSearchComponent] = None
        self.llm: Optional[LLMComponent] = None

        # Session management
        self.sessions: Dict[str, SessionState] = {}  # In-memory session store

        # Action handlers
        self.action_handlers: Dict[str, Callable] = {}

    async def initialize(self) -> None:
        """Initialize all sub-components"""
        if self._initialized:
            logger.info("Dialogue Manager already initialized")
            return

        logger.info("Initializing Dialogue Manager...")
        start_time = time.time()

        # Initialize NER component
        if self.config.enable_ner:
            ner_config = self.config.ner_config or NERConfig(device=self.config.device)
            self.ner = NERComponent(ner_config)
            await self.ner.initialize()
            logger.info("NER component initialized")

        # Initialize Classification component
        if self.config.enable_classification:
            cls_config = self.config.classification_config or ClassificationConfig(device=self.config.device)
            self.classifier = ClassificationComponent(cls_config)
            await self.classifier.initialize()
            logger.info("Classification component initialized")

        # Initialize Search component
        if self.config.enable_search:
            search_config = self.config.search_config or SemanticSearchConfig(device=self.config.device)
            self.searcher = SemanticSearchComponent(search_config)
            await self.searcher.initialize()
            logger.info("Search component initialized")

        # Initialize LLM component (optional)
        if self.config.enable_llm:
            llm_config = self.config.llm_config or LLMConfig(device=self.config.device)
            self.llm = LLMComponent(llm_config)
            await self.llm.initialize()
            logger.info("LLM component initialized")

        self._initialized = True
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Dialogue Manager initialized in {elapsed:.2f}ms")

    async def process(self, input_data: DialogueInput) -> DialogueOutput:
        """Process user input through dialogue pipeline"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # 1. Get or create session
            session = self._get_or_create_session(input_data.session_id)

            # 2. Add user message to history
            session.add_message(MessageRole.USER, input_data.text)

            # 3. Fragment text if needed
            fragments = self._fragment_text(input_data.text) if self.config.enable_text_fragmentation else [input_data.text]
            logger.info(f"Processing {len(fragments)} text fragment(s)")

            # 4. Check if query is fractional (context-dependent)
            is_fractional = await self._is_fractional_query(input_data.text, session)

            # 5. Augment with context if fractional
            processing_text = input_data.text
            if is_fractional and self.config.enable_context_augmentation:
                context = session.get_recent_context(self.config.max_context_turns)
                processing_text = f"{context}\nCurrent: {input_data.text}"
                logger.info("Augmented with conversation context")

            # 6. Extract entities
            entities = {}
            if input_data.extract_entities and self.config.enable_ner and self.ner:
                ner_result = await self.ner.process(NERInput(text=processing_text))
                if ner_result.success:
                    entities = ner_result.entities
                    session.update_entities(entities)
                    logger.info(f"Extracted entities: {list(entities.keys())}")

            # 7. Classify intent
            intent = None
            confidence = 0.0
            if input_data.classify_intent and self.config.enable_classification and self.classifier:
                cls_result = await self.classifier.process(ClassificationInput(text=processing_text))
                if cls_result.success:
                    intent = cls_result.top_cluster
                    confidence = cls_result.top_confidence
                    session.current_intent = intent
                    logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")

            # 8. Search knowledge base
            knowledge_results = []
            if input_data.search_knowledge and self.config.enable_search and self.searcher:
                search_result = await self.searcher.process(
                    SemanticSearchInput(text=processing_text, cluster=intent)
                )
                if search_result.success:
                    knowledge_results = [
                        {
                            "question": r.question,
                            "cluster": r.cluster,
                            "tag": r.tag,
                            "score": r.score,
                        }
                        for r in search_result.results
                    ]
                    logger.info(f"Found {len(knowledge_results)} knowledge results")

            # 9. Invoke actions (forms, APIs, etc.)
            actions = await self._invoke_actions(session, intent, entities)

            # 10. Generate response
            response = ""
            if input_data.generate_response:
                response = await self._generate_response(
                    session=session,
                    intent=intent,
                    entities=entities,
                    knowledge_results=knowledge_results,
                    actions=actions,
                )

            # 11. Add assistant response to history
            if response:
                session.add_message(MessageRole.ASSISTANT, response)

            processing_time_ms = (time.time() - start_time) * 1000

            return DialogueOutput(
                success=True,
                data={
                    "response": response,
                    "session_id": session.session_id,
                    "entities": session.entities,
                    "intent": intent,
                    "confidence": confidence,
                    "actions": actions,
                    "knowledge_results": knowledge_results,
                },
                response=response,
                session_id=session.session_id,
                entities=session.entities,
                intent=intent,
                confidence=confidence,
                actions=actions,
                knowledge_results=knowledge_results,
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

        except Exception as e:
            logger.error(f"Dialogue processing failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return DialogueOutput(
                success=False,
                data={},
                error=str(e),
                processing_time_ms=processing_time_ms,
                component_name=self.get_name(),
            )

    def _get_or_create_session(self, session_id: Optional[str]) -> SessionState:
        """Get existing session or create new one"""
        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
            logger.info(f"Created new session: {session_id}")
        else:
            logger.info(f"Retrieved existing session: {session_id}")

        return self.sessions[session_id]

    def _fragment_text(self, text: str) -> List[str]:
        """
        Fragment long text into chunks.
        Splits on sentence boundaries when possible.
        """
        if len(text) <= self.config.max_fragment_length:
            return [text]

        # Simple sentence splitting (can be enhanced)
        import re
        sentences = re.split(r'[।.!?]+', text)

        fragments = []
        current_fragment = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_fragment) + len(sentence) + 1 <= self.config.max_fragment_length:
                current_fragment += sentence + " "
            else:
                if current_fragment:
                    fragments.append(current_fragment.strip())
                current_fragment = sentence + " "

        if current_fragment:
            fragments.append(current_fragment.strip())

        return fragments

    async def _is_fractional_query(self, text: str, session: SessionState) -> bool:
        """
        Detect if query is fractional (context-dependent).

        Fractional queries are short, incomplete queries that depend on context.
        Examples: "এবং এটি?", "আরও বলুন", "কিভাবে?"
        """
        # Simple heuristics (can be enhanced with classifier)
        if len(text.split()) <= 3:
            return True

        # Check for context-dependent words (Bengali examples)
        context_markers = ["এটি", "এবং", "আরও", "কিভাবে", "কেন", "কোথায়"]
        if any(marker in text for marker in context_markers):
            return True

        # If no prior context, not fractional
        if len(session.messages) <= 1:
            return False

        return False

    async def _invoke_actions(
        self,
        session: SessionState,
        intent: Optional[str],
        entities: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Invoke registered action handlers based on intent and entities.

        Actions can be:
        - Form triggers
        - API calls
        - Database queries
        - External service invocations
        """
        actions = []

        for action_name, handler in self.action_handlers.items():
            try:
                result = await handler(session, intent, entities)
                if result:
                    actions.append({
                        "action": action_name,
                        "result": result,
                    })
            except Exception as e:
                logger.error(f"Action {action_name} failed: {e}")
                actions.append({
                    "action": action_name,
                    "error": str(e),
                })

        return actions

    async def _generate_response(
        self,
        session: SessionState,
        intent: Optional[str],
        entities: Dict[str, str],
        knowledge_results: List[Dict],
        actions: List[Dict],
    ) -> str:
        """
        Generate response using either:
        1. LLM (if enabled)
        2. Template-based (fallback)
        3. Knowledge base answer (if available)
        """
        # Use knowledge base answer if available
        if knowledge_results and len(knowledge_results) > 0:
            top_result = knowledge_results[0]
            if top_result.get("score", 0) > 0.8:
                # High confidence - return answer directly
                # (Would need answer lookup in real implementation)
                return f"Based on '{top_result['question']}', here's the answer..."

        # Use LLM if enabled
        if self.llm:
            prompt = self._build_llm_prompt(session, intent, entities, knowledge_results)
            llm_result = await self.llm.process(LLMInput(prompt=prompt))
            if llm_result.success:
                return llm_result.generated_text

        # Fallback: template-based response
        if intent:
            return f"I understand you're asking about {intent}. "

        return "I'm processing your request. How can I help you further?"

    def _build_llm_prompt(
        self,
        session: SessionState,
        intent: Optional[str],
        entities: Dict[str, str],
        knowledge_results: List[Dict],
    ) -> str:
        """Build prompt for LLM response generation"""
        context = session.get_recent_context(max_turns=2)

        prompt = f"""Given the conversation context and user query, provide a helpful response.

Context:
{context}

Intent: {intent or 'unknown'}
Entities: {entities}

Relevant Knowledge:
{knowledge_results[0]['question'] if knowledge_results else 'None'}

Generate a helpful response:"""

        return prompt

    def register_action_handler(self, action_name: str, handler: Callable):
        """Register an action handler"""
        self.action_handlers[action_name] = handler
        logger.info(f"Registered action handler: {action_name}")

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")

    def get_name(self) -> str:
        """Return component name"""
        return "dialogue_manager"

    async def health_check(self) -> Dict[str, Any]:
        """Health check for Dialogue Manager"""
        base_health = await super().health_check()

        dm_specific = {
            "ner_enabled": self.config.enable_ner,
            "classification_enabled": self.config.enable_classification,
            "search_enabled": self.config.enable_search,
            "llm_enabled": self.config.enable_llm,
            "active_sessions": len(self.sessions),
            "registered_actions": len(self.action_handlers),
        }

        return {**base_health, **dm_specific}


# CLI entry point
async def main():
    """CLI entry point for Dialogue Manager"""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Dialogue Manager")
    parser.add_argument("--text", type=str, help="User input text")
    parser.add_argument("--session-id", type=str, help="Session ID")
    parser.add_argument("--input-file", type=str, help="Input JSON file")
    parser.add_argument("--output-file", type=str, help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM response generation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Create config
    config = DialogueConfig(
        device=args.device,
        enable_llm=args.enable_llm,
    )

    # Initialize manager
    manager = DialogueManager(config)
    await manager.initialize()

    # Interactive mode
    if args.interactive:
        print("Dialogue Manager - Interactive Mode")
        print("Type 'exit' to quit\n")

        session_id = args.session_id or str(uuid.uuid4())

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            if not user_input:
                continue

            input_data = DialogueInput(text=user_input, session_id=session_id)
            output = await manager.process(input_data)

            if output.success:
                print(f"Bot: {output.response}")
                if output.entities:
                    print(f"  [Entities: {output.entities}]")
                if output.intent:
                    print(f"  [Intent: {output.intent} ({output.confidence:.2f})]")
            else:
                print(f"Error: {output.error}")

            print()

        sys.exit(0)

    # Single query mode
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_json = json.load(f)
            input_data = DialogueInput(**input_json)
    elif args.text:
        input_data = DialogueInput(text=args.text, session_id=args.session_id)
    else:
        print("Error: Must provide --text, --input-file, or --interactive", file=sys.stderr)
        sys.exit(1)

    # Process
    output = await manager.process(input_data)

    # Write output
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output.to_json())
    else:
        print(output.to_json())

    sys.exit(0 if output.success else 1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

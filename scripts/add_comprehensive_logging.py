#!/usr/bin/env python3
"""
Add comprehensive logging throughout the dialogue system for easy debugging.

This script enhances logging in:
1. Dialogue pipeline stages
2. Policy engine & handlers
3. NLP service
4. Form execution
5. Tenant context initialization
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def add_logging_to_dialogue_service():
    """Add comprehensive logging to dialogue_service.py"""
    file_path = PROJECT_ROOT / "src/application/dialogue_service.py"
    content = file_path.read_text()

    # Add pipeline start logging
    content = re.sub(
        r'(class SessionLoadStage.*?\n.*?async def process.*?\n.*?"""Load or create session""")',
        r'\1\n        logger.info(f"[PIPELINE] ========== STARTING PIPELINE ==========")\n        logger.info(f"[PIPELINE] Session: {context.session_id} | Query: \'{context.query}\'")\n        logger.info(f"[PIPELINE] User: {context.user_id} | Tenant: {context.tenant_id}")',
        content,
        flags=re.DOTALL
    )

    # Add session load logging
    content = re.sub(
        r'(context\.session_state = session_state\n        logger\.debug\(f"Session loaded:)',
        r'context.session_state = session_state\n        logger.info(f"[SESSION] Loaded session | Turn count: {session_state.turn_count} | Active form: {session_state.active_form or \'None\'}|")\n        logger.debug(f"[SESSION] History: {len(session_state.history)} turns")\n        logger.debug(f"Session loaded:',
        content
    )

    # Add NLP stage logging
    content = re.sub(
        r'(class NLPStage.*?async def process.*?\n.*?"""Run NLP pipeline""")',
        r'\1\n        logger.info(f"[PIPELINE] Stage: NLP Processing")\n        logger.debug(f"[FLOW] → NLPStage: Classification + Semantic Search")',
        content,
        flags=re.DOTALL
    )

    # Add policy stage logging
    content = re.sub(
        r'(class PolicyStage.*?async def process.*?\n.*?"""Policy decision""")',
        r'\1\n        logger.info(f"[PIPELINE] Stage: Policy Decision")\n        logger.debug(f"[FLOW] → PolicyStage: Routing through handler chain")',
        content,
        flags=re.DOTALL
    )

    file_path.write_text(content)
    print(f"✅ Enhanced logging in {file_path}")


def add_logging_to_policy_handlers():
    """Add logging to policy handlers"""
    file_path = PROJECT_ROOT / "src/application/policy/handlers.py"
    content = file_path.read_text()

    # Add handler execution logging
    content = re.sub(
        r'(async def handle\([\s\S]*?\) -> Optional\[Action\]:[\s\S]*?"""Handle request or pass to next handler\.""")',
        r'\1\n        logger.debug(f"[HANDLER] {self.__class__.__name__} processing...")',
        content
    )

    # ActiveFormHandler logging
    content = re.sub(
        r'(class ActiveFormHandler.*?async def _process.*?\n.*?"""If form is active.*?""")',
        r'\1\n        logger.debug(f"[HANDLER] ActiveFormHandler: active_form={session_state.active_form}")',
        content,
        flags=re.DOTALL
    )

    # HighConfidenceFAQHandler logging
    content = re.sub(
        r'(if confidence >= self\.faq_policy\.confidence_threshold:)',
        r'logger.info(f"[HANDLER] HighConfidenceFAQHandler: confidence={confidence:.3f} >= threshold={self.faq_policy.confidence_threshold:.3f}")\n        \1',
        content
    )

    file_path.write_text(content)
    print(f"✅ Enhanced logging in {file_path}")


def add_logging_to_nlp_service():
    """Add logging to NLP service"""
    file_path = PROJECT_ROOT / "src/application/nlp_service.py"
    content = file_path.read_text()

    # Add NLP processing start
    content = re.sub(
        r'(async def process\(self, query: str\).*?\n.*?"""Process query through NLP pipeline""")',
        r'\1\n        logger.info(f"[NLP] Processing query: \'{query[:60]}...\'")  \n        logger.debug(f"[NLP] Full query length: {len(query)} chars")',
        content,
        flags=re.DOTALL
    )

    file_path.write_text(content)
    print(f"✅ Enhanced logging in {file_path}")


def add_logging_to_form_runner():
    """Add logging to form execution"""
    file_path = PROJECT_ROOT / "src/application/forms/form_runner.py"
    if not file_path.exists():
        print(f"⚠️  Skipping {file_path} (not found)")
        return

    content = file_path.read_text()

    # Add form start logging
    content = re.sub(
        r'(async def collect_slot\([\s\S]*?\):[\s\S]*?"""Collect a single slot""")',
        r'\1\n        logger.info(f"[FORM] Collecting slot: {slot_name} from form: {form.__class__.__name__}")\n        logger.debug(f"[FORM] Current state: filled={list(state.filled_slots.keys())}")',
        content
    )

    file_path.write_text(content)
    print(f"✅ Enhanced logging in {file_path}")


def add_logging_to_tenant_context():
    """Add logging to tenant context initialization"""
    file_path = PROJECT_ROOT / "src/shared/tenant_context.py"
    content = file_path.read_text()

    # Add NLP service initialization logging
    content = re.sub(
        r'(async def get_nlp_service\(self\).*?if self\._nlp_service is None:)',
        r'\1\n                logger.info(f"[CONTEXT] Initializing NLP service for tenant: {self.config.tenant_id}")',
        content,
        flags=re.DOTALL
    )

    # Add policy engine initialization logging
    content = re.sub(
        r'(async def get_policy_engine\(self\).*?if self\._policy_engine is None:)',
        r'\1\n                logger.info(f"[CONTEXT] Initializing policy engine for tenant: {self.config.tenant_id}")',
        content,
        flags=re.DOTALL
    )

    file_path.write_text(content)
    print(f"✅ Enhanced logging in {file_path}")


def create_logging_config():
    """Create enhanced logging configuration"""
    config_path = PROJECT_ROOT / "config/logging_config.json"

    config = """{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "detailed": {
      "format": "%(asctime)s | %(levelname)-8s | [%(name)s:%(lineno)d] | %(message)s",
      "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    "simple": {
      "format": "%(levelname)-8s | %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "INFO",
      "formatter": "detailed",
      "stream": "ext://sys.stdout"
    },
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "DEBUG",
      "formatter": "detailed",
      "filename": "logs/dialogue_system.log",
      "maxBytes": 10485760,
      "backupCount": 5
    }
  },
  "loggers": {
    "src.application.dialogue_service": {
      "level": "INFO",
      "handlers": ["console", "file"],
      "propagate": false
    },
    "src.application.policy": {
      "level": "INFO",
      "handlers": ["console", "file"],
      "propagate": false
    },
    "src.application.nlp_service": {
      "level": "INFO",
      "handlers": ["console", "file"],
      "propagate": false
    },
    "src.application.forms": {
      "level": "INFO",
      "handlers": ["console", "file"],
      "propagate": false
    },
    "src.shared.tenant_context": {
      "level": "INFO",
      "handlers": ["console", "file"],
      "propagate": false
    }
  },
  "root": {
    "level": "INFO",
    "handlers": ["console", "file"]
  }
}
"""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config)
    print(f"✅ Created logging config: {config_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Adding Comprehensive Logging to Dialogue System")
    print("=" * 70)
    print()

    try:
        add_logging_to_dialogue_service()
        add_logging_to_policy_handlers()
        add_logging_to_nlp_service()
        add_logging_to_form_runner()
        add_logging_to_tenant_context()
        create_logging_config()

        print()
        print("=" * 70)
        print("✅ Logging enhancement complete!")
        print("=" * 70)
        print()
        print("Logging prefixes added:")
        print("  [PIPELINE]     - Pipeline stage execution")
        print("  [SESSION]      - Session management")
        print("  [NLP]          - NLP processing")
        print("  [POLICY]       - Policy decisions")
        print("  [HANDLER]      - Handler chain execution")
        print("  [FORM]         - Form execution")
        print("  [CONTEXT]      - Tenant context initialization")
        print("  [FLOW]         - Overall flow tracking")
        print()
        print("To see detailed logs, set LOG_LEVEL=DEBUG when running the server:")
        print("  LOG_LEVEL=DEBUG uvicorn src.interfaces.api.app:app --reload")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""Regex patterns for structured entity extraction."""

import re
from typing import Dict, List, Optional


class EntityPatterns:
    """
    Regex-based entity extraction for structured data.
    
    Supports: NID numbers, account numbers, phone numbers, email, dates
    """

    # Structured ID patterns
    NID_PATTERN = r'\b\d{17}\b'
    ACCOUNT_PATTERN = r'\b\d{10,16}\b'
    PHONE_PATTERN = r'\b0?1[3-9]\d{8}\b'
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Date patterns (multiple formats)
    DATE_PATTERNS = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
    ]

    NUMBER_PATTERN = r'\b\d+\b'

    def __init__(self):
        """Initialize pattern matchers."""
        self.patterns = {
            'nid_number': re.compile(self.NID_PATTERN),
            'account_number': re.compile(self.ACCOUNT_PATTERN),
            'phone': re.compile(self.PHONE_PATTERN),
            'email': re.compile(self.EMAIL_PATTERN, re.IGNORECASE),
        }
        self.date_matchers = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self.number_matcher = re.compile(self.NUMBER_PATTERN)

    def extract_all(self, text: str) -> Dict[str, List[str]]:
        """Extract all structured entities from text."""
        entities = {}

        # Extract structured IDs
        for entity_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = matches

        # Extract dates
        date_matches = []
        for date_matcher in self.date_matchers:
            date_matches.extend(date_matcher.findall(text))
        if date_matches:
            entities['date'] = date_matches

        # Extract numbers
        number_matches = self.number_matcher.findall(text)
        if number_matches:
            filtered_numbers = [
                n for n in number_matches
                if not any(n in str(v) for values in entities.values() for v in values)
            ]
            if filtered_numbers:
                entities['number'] = filtered_numbers

        return entities

    def extract_entity_type(self, text: str, entity_type: str) -> Optional[str]:
        """Extract specific entity type from text."""
        if entity_type == 'date':
            for matcher in self.date_matchers:
                match = matcher.search(text)
                if match:
                    return match.group(0)
            return None

        pattern = self.patterns.get(entity_type)
        if pattern:
            match = pattern.search(text)
            return match.group(0) if match else None

        return None

    def validate_entity(self, value: str, entity_type: str) -> bool:
        """Validate if a value matches the expected pattern."""
        if entity_type == 'date':
            return any(matcher.match(value) for matcher in self.date_matchers)

        pattern = self.patterns.get(entity_type)
        if pattern:
            return pattern.match(value) is not None

        return False

    def prioritize_entities(self, entities: Dict[str, List[str]]) -> Dict[str, str]:
        """Prioritize entities when multiple of same type found."""
        prioritized = {}

        for entity_type, matches in entities.items():
            if matches:
                prioritized[entity_type] = matches[0]

        # Resolve conflicts: NID takes precedence over account
        if 'nid_number' in prioritized and 'account_number' in prioritized:
            if len(prioritized['nid_number']) == 17:
                if prioritized['account_number'] == prioritized['nid_number']:
                    del prioritized['account_number']

        return prioritized


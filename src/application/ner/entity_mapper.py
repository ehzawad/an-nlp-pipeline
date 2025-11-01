"""Entity label to slot name mapping."""

from typing import Dict, List, Optional


class EntityMapper:
    """Maps generic NER labels to domain-specific slot names."""

    # Standard NER labels to generic entity types
    NER_LABEL_MAP = {
        "PER": "person_name",
        "PERSON": "person_name",
        "B-PER": "person_name",
        "I-PER": "person_name",
        "LOC": "location",
        "LOCATION": "location",
        "GPE": "location",
        "B-LOC": "location",
        "I-LOC": "location",
        "ORG": "organization",
        "ORGANIZATION": "organization",
        "B-ORG": "organization",
        "I-ORG": "organization",
        "DATE": "date",
        "B-DATE": "date",
        "I-DATE": "date",
        "MISC": "misc",
        "B-MISC": "misc",
        "I-MISC": "misc",
    }

    # Generic entity types to potential slot names
    ENTITY_TO_SLOTS = {
        "person_name": [
            "full_name", "name", "customer_name", "first_name",
            "last_name", "passenger_name", "guest_name",
        ],
        "location": [
            "city", "address", "district", "location",
            "destination", "origin", "delivery_address",
        ],
        "date": [
            "date_of_birth", "dob", "check_in_date", "check_out_date",
            "appointment_date", "booking_date", "travel_date", "date",
        ],
        "number": [
            "num_guests", "quantity", "count", "amount",
            "number_of_rooms", "number_of_travelers",
        ],
        "account_number": ["account_number", "account_id", "account"],
        "nid_number": ["nid_number", "nid", "national_id"],
        "phone": ["phone", "phone_number", "mobile", "mobile_number", "contact_number"],
        "email": ["email", "email_address"],
    }

    def __init__(self):
        """Initialize mapper with label mappings."""
        self.ner_label_map = self.NER_LABEL_MAP
        self.entity_to_slots = self.ENTITY_TO_SLOTS

    def map_ner_label(self, ner_label: str) -> Optional[str]:
        """Map NER model label to generic entity type."""
        return self.ner_label_map.get(ner_label)

    def find_matching_slots(
        self,
        entity_type: str,
        available_slots: List[str]
    ) -> List[str]:
        """Find which slots match the entity type."""
        potential_slots = self.entity_to_slots.get(entity_type, [])
        matches = [slot for slot in available_slots if slot in potential_slots]
        return matches

    def map_entities_to_slots(
        self,
        extracted_entities: Dict[str, str],
        form_slots: List[str]
    ) -> Dict[str, str]:
        """Map extracted entities to form slots."""
        mapped = {}

        for entity_type, entity_value in extracted_entities.items():
            matching_slots = self.find_matching_slots(entity_type, form_slots)

            if matching_slots:
                slot_name = matching_slots[0]
                mapped[slot_name] = entity_value

        return mapped

    def get_slot_entity_type(self, slot_name: str) -> Optional[str]:
        """Get entity type that matches a slot name (reverse mapping)."""
        for entity_type, slot_list in self.entity_to_slots.items():
            if slot_name in slot_list:
                return entity_type
        return None


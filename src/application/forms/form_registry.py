"""Form registry for centralized form management."""

from typing import Dict, List, Optional
from .base_form import BaseForm


class FormRegistry:
    """
    Registry for all available forms.

    Forms register themselves here to be discovered by the dialogue system.
    """

    def __init__(self):
        self._forms: Dict[str, BaseForm] = {}
        self._tag_to_form: Dict[str, str] = {}

    def register(self, form: BaseForm):
        """
        Register a form.

        Args:
            form: Form instance to register
        """
        form_name = form.form_name
        if form_name in self._forms:
            raise ValueError(f"Form '{form_name}' already registered")

        self._forms[form_name] = form

        # Build tag → form_name mapping
        for tag in form.trigger_tags:
            if tag in self._tag_to_form:
                existing = self._tag_to_form[tag]
                raise ValueError(
                    f"Tag '{tag}' already mapped to form '{existing}'. "
                    f"Cannot map to '{form_name}' as well."
                )
            self._tag_to_form[tag] = form_name

    def get(self, form_name: str) -> Optional[BaseForm]:
        """Get form by name."""
        return self._forms.get(form_name)

    def get_by_tag(self, tag: str) -> Optional[BaseForm]:
        """Get form by trigger tag."""
        form_name = self._tag_to_form.get(tag)
        if form_name:
            return self._forms.get(form_name)
        return None

    def list_forms(self) -> List[str]:
        """List all registered form names."""
        return list(self._forms.keys())

    def __len__(self) -> int:
        """Number of registered forms."""
        return len(self._forms)

    def __contains__(self, form_name: str) -> bool:
        """Check if form is registered."""
        return form_name in self._forms


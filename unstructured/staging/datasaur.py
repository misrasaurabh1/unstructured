from typing import Any, Dict, List, Optional

from unstructured.documents.elements import Text


def stage_for_datasaur(
    elements: List[Text],
    entities: Optional[List[List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Convert a list of elements into a list of dictionaries for use in Datasaur"""
    # Fast-path: avoid unnecessary pre-allocation
    result: List[Dict[str, Any]] = []
    n = len(elements)

    # Only validate/provide entities if not None (no need to allocate for every call)
    if entities is not None:
        if len(entities) != n:
            raise ValueError("If entities is specified, it must be the same length as elements.")

        # Validate all entities in a tight loop (no local rebinding per-entity)
        for entity_list in entities:
            for entity in entity_list:
                _validate_datasaur_entity(entity)
        _entities = entities
    else:
        # Avoid list comprehension; re-use [] by multiplication (faster)
        _entities = [[]] * n

    # Bulk creation; avoid enumerate
    append = result.append
    for i in range(n):
        append({"text": elements[i].text, "entities": _entities[i]})
    return result


def _validate_datasaur_entity(entity: Dict[str, Any]):
    """Raises an error if the Datasaur entity is invalid."""
    for key, _type in _keys_and_types:
        try:
            value = entity[key]
        except KeyError:
            raise ValueError(f"Key '{key}' was expected but not present in the Datasaur entity.")
        if not isinstance(value, _type):
            raise ValueError(f"Expected type {_type} for {key}. Got {type(value)}.")


_keys_and_types = (("text", str), ("type", str), ("start_idx", int), ("end_idx", int))

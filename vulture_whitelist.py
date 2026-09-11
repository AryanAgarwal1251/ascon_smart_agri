"""Vulture whitelist: intentional false positives.

Vulture flags code that looks unused. Some of our code is legitimately "unused" from a
static point of view because a framework calls it, or because it is a public API entry
point. List those names here so vulture does not report them. Referencing an attribute on
the throwaway ``_whitelist`` object is the idiomatic way to mark a name as used.

During scaffolding, ``min_confidence = 80`` (see pyproject.toml) already suppresses the
"defined-but-not-yet-called" stub functions/classes (confidence 60). This file is for the
higher-confidence cases that survive that filter, and grows as real code lands.
"""


class _Whitelist:
    def __getattr__(self, _name: str) -> None:
        return None


_whitelist = _Whitelist()

# PyTorch calls Module.forward() via __call__, so it is never referenced directly.
_whitelist.forward

# Pydantic field validators are invoked by the framework, not by our code.
_whitelist._classes_is_eight
_whitelist._at_least_three_seeds

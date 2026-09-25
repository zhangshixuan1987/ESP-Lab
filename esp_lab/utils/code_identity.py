"""Function-level code fingerprints for cache provenance.

Caches that must be invalidated when their computing code changes should not
hash whole source files: an unrelated edit (a new helper, a docstring, a
filename tweak) would then force an expensive rebuild. :func:`code_digest`
hashes only the given functions and everything they call inside the project
packages, followed recursively, from a comment- and docstring-free parse.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import sys
import textwrap
import types
from typing import Any, Iterable

PROJECT_PACKAGES = ("esp_lab", "workflows")


def _in_project(module_name: str | None) -> bool:
    return bool(module_name) and module_name.split(".")[0] in PROJECT_PACKAGES


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant
            ) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _normalized(source: str) -> str:
    """Return a formatting-, comment- and docstring-insensitive form of ``source``."""
    tree = _strip_docstrings(ast.parse(textwrap.dedent(source)))
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def _local_namespace(tree: ast.AST, module: types.ModuleType) -> dict[str, Any]:
    """Resolve imports made inside a function body (e.g. deferred imports)."""
    names: dict[str, Any] = {}
    for node in ast.walk(tree):
        try:
            if isinstance(node, ast.ImportFrom) and (node.module or node.level):
                target = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), module.__package__ or module.__name__
                ) if node.level else node.module
                parent = importlib.import_module(target)
                for alias in node.names:
                    value = getattr(parent, alias.name, None)
                    if value is None:
                        value = importlib.import_module(f"{target}.{alias.name}")
                    names[alias.asname or alias.name] = value
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported = importlib.import_module(alias.name)
                    names[alias.asname or alias.name.split(".")[0]] = (
                        imported if alias.asname else importlib.import_module(alias.name.split(".")[0])
                    )
        except ImportError:
            continue
    return names


def _module_constant_source(module: types.ModuleType, name: str) -> str | None:
    """Return the source of a top-level assignment to ``name`` in ``module``."""
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return None
    for node in ast.parse(source).body:
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.get_source_segment(source, node)
    return None


def code_digest(objects: Iterable[Any], *, length: int = 64) -> str:
    """Hash ``objects`` and the project code they reach, ignoring formatting.

    ``objects`` are functions or classes. Every name they use that resolves to a
    project function, class, or module-level constant is included, recursively.
    Third-party code (numpy, xarray, ...) is not hashed; pin it through
    algorithm version strings instead.
    """
    parts: dict[str, str] = {}
    stack = list(objects)
    while stack:
        obj = inspect.unwrap(stack.pop())
        module = sys.modules.get(getattr(obj, "__module__", None) or "")
        if module is None or not _in_project(module.__name__):
            continue
        key = f"{module.__name__}.{getattr(obj, '__qualname__', obj)}"
        if key in parts:
            continue
        try:
            source = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        parts[key] = _normalized(source)
        tree = ast.parse(textwrap.dedent(source))
        namespace = {**vars(module), **_local_namespace(tree, module)}
        for node in ast.walk(tree):
            value, name = None, None
            if isinstance(node, ast.Name):
                name = node.id
                value = namespace.get(name)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                base = namespace.get(node.value.id)
                if isinstance(base, types.ModuleType):
                    name, value = node.attr, getattr(base, node.attr, None)
                    module_for_const = base
            if value is None:
                continue
            if inspect.isfunction(value) or inspect.isclass(value):
                if _in_project(getattr(value, "__module__", None)):
                    stack.append(value)
            elif not isinstance(value, types.ModuleType) and name:
                owner = module_for_const if isinstance(node, ast.Attribute) else module
                if _in_project(owner.__name__):
                    const = _module_constant_source(owner, name)
                    if const is not None:
                        parts[f"{owner.__name__}.{name}"] = _normalized(const)
    payload = "\n".join(f"{key}\n{parts[key]}" for key in sorted(parts))
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


__all__ = ["code_digest"]

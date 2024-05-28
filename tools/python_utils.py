# Tools for converting python2 to python3
import re
from lib2to3.refactor import (
    RefactoringTool,
    get_fixers_from_package,
    _detect_future_features,
)
from lib2to3.pgen2.driver import Driver
from lib2to3.pygram import python_grammar
from lib2to3 import pytree
from langchain_core.tools import tool


def _sanitize(code: str) -> str:
    """Sanitize code before converting to python3.
    Replace words that start with $ by their version without the __ instead of $.
    Add a trailing newline to the code (otherwise lib2to3 will fail)
    Don't alter words that have $ in the middle or at the end.
    """
    new_code = re.sub(r"\$(?=[a-zA-Z])", "__", code)
    new_code = new_code.rstrip() + "\n"
    return new_code


def _desanitize(code: str) -> str:
    """Desanitize code after converting to python3.
    Replace words that start with __ by their version with the $ instead of __.
    Remove trailing newline from the code (added during sanitization).
    Don't alter words that have __ in the middle or at the end.
    """
    new_code = code.rstrip("\n")
    return re.sub(r"__(?=[a-zA-Z])", "$", new_code)


@tool
def migrate_to_py3(code):
    """Convert code from python2 to python3"""
    """This should only be used for code that is already valid python2 code and not for python3 code"""
    driver = Driver(grammar=python_grammar, convert=pytree.convert)
    fixers = get_fixers_from_package("lib2to3.fixes")
    refactor_tool = RefactoringTool(fixers)
    sanitized = _sanitize(code)
    tree = driver.parse_string(sanitized)
    tree.future_features = _detect_future_features(sanitized)
    refactor_tool.refactor_tree(tree, "")
    python3_code = _desanitize(str(tree))
    return python3_code

"""Minimal pyext shim for verl prime_code (py3.12-compatible).
Provides RuntimeModule.from_string, the only symbol prime_code/testing_util uses.
Executes candidate solutions in an ephemeral module namespace — same semantics
as the original APPS/PRIME evaluation harness."""
import types


class RuntimeModule:
    @staticmethod
    def from_string(name, docstring, code):
        mod = types.ModuleType(name, docstring)
        exec(code, mod.__dict__)
        return mod

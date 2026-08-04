"""
SANDBOXING TOOL EXECUTION & LEAST-PRIVILEGE TOOLS
===================================================
Agents with raw file or shell tool access can compromise host systems 
(e.g., executing command injections like `rm -rf /` or spawning infinite loops).

This script demonstrates secure tool practices:
1. Argument validation and sanitization whitelists.
2. Timeout-bounded subprocess executing inside sandboxed environments.
"""

import subprocess
import re
from langchain_core.tools import tool

# --------------------------------------------------------------------------
# Technique 1: Input Validation and Shell Injection Defenses
# --------------------------------------------------------------------------
@tool
def read_workspace_file(filename: str) -> str:
    """Reads a file in the workspace directory. Only allows safe alphanumeric basenames."""
    # Defend against path traversal attacks (e.g., '../../etc/passwd')
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("Security violation: path traversal terms are forbidden.")
        
    # Match strictly safe characters (alphanumeric, dot, dash, underscore)
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', filename):
        raise ValueError("Security violation: filename contains invalid characters.")
        
    return f"[MOCK CONTENT] Content of safe file: {filename}"

# Test cases
try:
    print(read_workspace_file.invoke({"filename": "../../secrets.json"}))
except ValueError as e:
    print(f"Path Traversal Blocked: {e}")

try:
    print(read_workspace_file.invoke({"filename": "some_config.yaml; rm -rf /"}))
except ValueError as e:
    print(f"Command Injection Blocked: {e}")


# --------------------------------------------------------------------------
# Technique 2: Subprocess Isolation and Timeout Controls
# --------------------------------------------------------------------------
@tool
def run_sandboxed_code(code: str) -> str:
    """Runs a Python code snippet inside a separate sandboxed process with resource timeouts."""
    # 1. Inspect code for forbidden modules to implement quick blocking
    forbidden_terms = ["import os", "import sys", "subprocess", "import socket", "eval", "exec"]
    for term in forbidden_terms:
        if term in code:
            return f"Security Error: Module or statement '{term}' is blocked."
            
    # 2. Run code in isolated process with strict timeout limit
    # We pass code to a subprocess Python runner.
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=2.0  # Safeguards against infinite loops
        )
        if result.returncode != 0:
            return f"Runtime Error:\n{result.stderr}"
        return f"Output:\n{result.stdout}"
    except subprocess.TimeoutExpired:
        return "Error: Code execution exceeded maximum allowable timeout (2 seconds)."

# Test infinite loop defense
print("\nTesting infinite loop code run:")
infinite_loop_code = "while True: pass"
print(run_sandboxed_code.invoke({"code": infinite_loop_code}))

# Test successful run
print("\nTesting simple math run:")
math_code = "print(2**10)"
print(run_sandboxed_code.invoke({"code": math_code}))

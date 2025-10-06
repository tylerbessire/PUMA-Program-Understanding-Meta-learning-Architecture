# extract_patterns.py
import json
import re
from collections import Counter

def extract_task_methods(code):
    """Extract all task.method() calls"""
    return re.findall(r'task\.(\w+)', code)

def extract_grid_methods(code):
    """Extract all grid.method() calls"""  
    return re.findall(r'(?:input|output|grid)\.(\w+)', code)

def categorize_solution(code):
    """Categorize by complexity"""
    if 'task.build_grid' in code or 'task.input_height' in code:
        return 'hallucinated'
    elif 'find_objects' in code or 'copy_grid' in code:
        return 'minimal_dsl'
    elif 'task[' in code or 'np.' in code:
        return 'pure_python'
    return 'unknown'

# Parse masterlog
with open('PUMA ARC 2025/artifacts/master_solution_log.jsonl') as f:
    solutions = [json.loads(line) for line in f if json.loads(line).get('status') == 'generated']

categories = Counter()
task_methods = Counter()
grid_methods = Counter()

for sol in solutions:
    code = sol.get('dsl_program', '')
    categories[categorize_solution(code)] += 1
    task_methods.update(extract_task_methods(code))
    grid_methods.update(extract_grid_methods(code))

print(f"Pure Python: {categories['pure_python']}")
print(f"Minimal DSL: {categories['minimal_dsl']}")
print(f"Hallucinated: {categories['hallucinated']}")
print(f"\nTop 10 task methods: {task_methods.most_common(10)}")
print(f"Top 10 grid methods: {grid_methods.most_common(10)}")

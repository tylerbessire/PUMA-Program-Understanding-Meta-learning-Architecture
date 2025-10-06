#!/usr/bin/env python3
"""Quick test of LLM loading and generation."""

import sys
import time
sys.path.insert(0, 'PUMA')

print('='*80)
print('LLM QUICK TEST')
print('='*80)

from arc_solver.llm_interface import LLMInterface, LLMConfig

print('\n1. Creating LLM interface...')
config = LLMConfig(model_name='microsoft/Phi-3-mini-4k-instruct', use_4bit=True)
llm = LLMInterface(config)

print('\n2. Loading model (this will take ~2 minutes on macOS)...')
start = time.time()
llm._lazy_load()
load_time = time.time() - start
print(f'   ✓ Loaded in {load_time:.1f}s')
print(f'   ✓ Initialized: {llm._initialized}')

if not llm._initialized:
    print('\n✗ LLM failed to initialize')
    sys.exit(1)

print('\n3. Testing generation...')
test_cases = [
    ('You are a helpful assistant.', 'What is 2+2? Answer in one word.'),
    ('You are an ARC puzzle expert.', 'What color is a red square?'),
]

for i, (sys_prompt, user_prompt) in enumerate(test_cases, 1):
    print(f'\n   Test {i}: {user_prompt}')
    start = time.time()
    response = llm.generate(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=20
    )
    gen_time = time.time() - start
    print(f'   Response ({gen_time:.1f}s): {response}')

print('\n' + '='*80)
print('✓ LLM TEST COMPLETE')
print('='*80)

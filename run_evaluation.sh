#!/bin/bash

# ARC Challenge Evaluation Runner
# Evaluate first 20 tasks with detailed logging and real-time progress

echo "🏆 ARC Challenge Evaluation System"
echo "=================================="
echo ""
echo "Choose evaluation mode:"
echo "1) Terminal mode (detailed logging, real-time)"
echo "2) GUI mode (visual interface)"
echo "3) Quick test (first 2 tasks only)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting terminal evaluation..."
        echo "📝 Logs will be saved to evaluation_logs/"
        echo ""
        python3 evaluate_first_20.py
        ;;
    2)
        echo ""
        echo "🖥️  Starting GUI evaluation..."
        echo "💡 Click 'Start Evaluation' in the GUI window"
        echo ""
        python3 evaluate_gui.py
        ;;
    3)
        echo ""
        echo "⚡ Quick test mode (first 2 tasks)..."
        echo ""
        python3 -c "
import json
import time
from evaluate_first_20 import load_data, grids_equal, to_grid
from arc_solver.solver import solve_task

challenges, solutions = load_data()
task_ids = list(challenges.keys())[:2]
total_score = 0

print('Task ID          | Status | Score | Accuracy | Details')
print('-' * 60)

for task_id in task_ids:
    start_time = time.time()
    try:
        task = challenges[task_id]
        result = solve_task(task)
        prediction = result['attempt_1'][0]
        gold_solution = solutions[task_id][0]
        
        is_correct, details, accuracy = grids_equal(prediction, gold_solution)
        score = 1 if is_correct else 0
        status = '✅ PASS' if is_correct else '❌ FAIL'
        total_score += score
        
    except Exception as e:
        score = 0
        status = '⚠️ ERROR'
        accuracy = 0.0
        details = str(e)[:30]
    
    duration = time.time() - start_time
    print(f'{task_id:15} | {status:6} | {score:5} | {accuracy*100:6.1f}% | {details[:30]}')

print(f'\\nQuick Test Score: {total_score}/2 = {total_score/2*100:.0f}%')
        "
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✨ Evaluation complete!"
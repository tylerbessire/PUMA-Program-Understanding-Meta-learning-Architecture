"""
PUMA ARC Solver - Enhanced Kaggle Notebook Code (v2.0)
Complete working code for Kaggle ARC competition with fixed submission logic
"""

# Step 1: Setup and extract the enhanced solver
from kaggle_setup_enhanced import setup_puma_solver, get_solver, run_kaggle_evaluation_enhanced

# Step 2: Extract and setup (run this once)
setup_puma_solver()

# Step 3: Enhanced usage with fixed submission formatting
def solve_task_enhanced(task):
    """Drop-in replacement with enhanced submission formatting."""
    
    # Get solver instance (cached after first call)
    if not hasattr(solve_task_enhanced, '_solver'):
        solve_task_enhanced._solver = get_solver(memory_optimized=True)
    
    # Solve the task
    result = solve_task_enhanced._solver.solve_task(task)
    
    # Enhanced submission formatting
    from kaggle_setup_enhanced import format_submission_entry_enhanced
    task_id = str(task.get('id', 'unknown'))
    formatted_result = format_submission_entry_enhanced(result, task_id)
    
    # Add to checkpoint tracking
    solve_task_enhanced._solver.add_submission_result(task_id, result)
    
    return formatted_result

# Step 4: Complete enhanced evaluation function
def solve_arc_tasks_with_puma_enhanced(debug_mode=False):
    """Complete enhanced PUMA evaluation with fixed submission format."""
    
    # Use the test path from your notebook
    test_path = TEST_PATH if 'TEST_PATH' in globals() else None
    
    # Run complete evaluation with enhanced formatting and checkpointing
    return run_kaggle_evaluation_enhanced(test_path, debug_mode=debug_mode)

# Step 5: Enhanced validation function
def validate_submission_enhanced(submission):
    """Enhanced validation with detailed feedback."""
    from kaggle_setup_enhanced import validate_submission_format
    
    print("🔍 Enhanced Submission Validation")
    print("=" * 50)
    
    # Basic format validation
    format_valid = validate_submission_format(submission)
    
    if format_valid:
        print("✅ Format validation PASSED")
        
        # Additional analysis
        total_tasks = len(submission)
        total_test_cases = sum(len(entry) for entry in submission.values())
        tasks_with_multiple_tests = sum(1 for entry in submission.values() if len(entry) > 1)
        
        print(f"📊 Analysis:")
        print(f"   - Total tasks: {total_tasks}")
        print(f"   - Total test cases: {total_test_cases}")
        print(f"   - Tasks with multiple test cases: {tasks_with_multiple_tests}")
        print(f"   - Average test cases per task: {total_test_cases/total_tasks:.2f}")
        
        # Sample validation
        sample_task = list(submission.keys())[0]
        sample_entry = submission[sample_task]
        print(f"📝 Sample task {sample_task}: {len(sample_entry)} test case(s)")
        
        return True
    else:
        print("❌ Format validation FAILED")
        return False

# Example enhanced usage:
if __name__ == "__main__":
    print("🔥 Running with Enhanced PUMA solver v2.0...")
    print("✨ Features:")
    print("   - Fixed submission format handling")
    print("   - Enhanced validation and debugging")
    print("   - Improved error handling")
    print("   - Better memory management")
    
    # Option 1: Debug mode (3 tasks)
    # submission = solve_arc_tasks_with_puma_enhanced(debug_mode=True)
    
    # Option 2: Full evaluation
    # submission = solve_arc_tasks_with_puma_enhanced(debug_mode=False)
    
    # validate_submission_enhanced(submission)
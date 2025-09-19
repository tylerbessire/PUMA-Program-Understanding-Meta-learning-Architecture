#!/usr/bin/env python3
"""
GUI-based ARC evaluation with real-time progress visualization.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import numpy as np
import time
import threading
from datetime import datetime
from pathlib import Path
import queue
import sys

# Import the core evaluation logic
from evaluate_first_20 import load_data, grids_equal, to_grid

class ARCEvaluationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ARC Challenge Evaluator - First 20 Tasks")
        self.root.geometry("1000x700")
        
        # Initialize attributes first
        self.evaluation_running = False
        self.results = []
        self.message_queue = queue.Queue()
        
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the GUI components."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏆 ARC Challenge Evaluation", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Evaluation", 
                                      command=self.start_evaluation)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                     command=self.stop_evaluation, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           maximum=20, length=200)
        self.progress_bar.grid(row=0, column=2, padx=(10, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to start evaluation")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.grid(row=1, column=0, columnspan=3, pady=(5, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Real-time Results", padding="5")
        results_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results treeview
        columns = ("Task ID", "Status", "Score", "Accuracy", "Time", "Details")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=10)
        
        # Configure columns
        self.tree.heading("Task ID", text="Task ID")
        self.tree.heading("Status", text="Status")
        self.tree.heading("Score", text="Score")
        self.tree.heading("Accuracy", text="Accuracy")
        self.tree.heading("Time", text="Time (s)")
        self.tree.heading("Details", text="Details")
        
        self.tree.column("Task ID", width=100)
        self.tree.column("Status", width=80)
        self.tree.column("Score", width=60)
        self.tree.column("Accuracy", width=80)
        self.tree.column("Time", width=80)
        self.tree.column("Details", width=300)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for treeview
        tree_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Summary frame
        summary_frame = ttk.LabelFrame(main_frame, text="Summary", padding="5")
        summary_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.summary_text = tk.Text(summary_frame, height=6, wrap=tk.WORD)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        summary_frame.columnconfigure(0, weight=1)
        
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
        summary_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        # Start message queue processing
        self.process_queue()
        
    def process_queue(self):
        """Process messages from the evaluation thread."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.handle_message(message)
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_queue)
        
    def handle_message(self, message):
        """Handle a message from the evaluation thread."""
        msg_type = message.get("type")
        
        if msg_type == "progress":
            self.progress_var.set(message["value"])
            self.status_var.set(message["status"])
            
        elif msg_type == "result":
            data = message["data"]
            
            # Add to treeview
            status_emoji = "✅" if data["status"] == "PASS" else "❌" if data["status"] == "FAIL" else "⚠️"
            self.tree.insert("", "end", values=(
                data["task_id"],
                f"{status_emoji} {data['status']}",
                data["score"],
                f"{data['accuracy']*100:.1f}%",
                f"{data['duration']:.1f}",
                data["details"][:50] + ("..." if len(data["details"]) > 50 else "")
            ))
            
            # Auto-scroll to bottom
            children = self.tree.get_children()
            if children:
                self.tree.see(children[-1])
                
        elif msg_type == "summary":
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, message["text"])
            
        elif msg_type == "complete":
            self.evaluation_running = False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_var.set("Evaluation complete!")
            
    def start_evaluation(self):
        """Start the evaluation in a separate thread."""
        if self.evaluation_running:
            return
            
        self.evaluation_running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.summary_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        
        # Start evaluation thread
        thread = threading.Thread(target=self.run_evaluation, daemon=True)
        thread.start()
        
    def stop_evaluation(self):
        """Stop the evaluation."""
        self.evaluation_running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Evaluation stopped by user")
        
    def run_evaluation(self):
        """Run the evaluation (called in separate thread)."""
        try:
            # Import solver
            from arc_solver.solver import solve_task
            
            # Load data
            self.message_queue.put({"type": "progress", "value": 0, "status": "Loading data..."})
            challenges, solutions = load_data()
            
            # Get first 20 tasks
            task_ids = list(challenges.keys())[:20]
            total_score = 0
            start_time = time.time()
            
            for i, task_id in enumerate(task_ids):
                if not self.evaluation_running:
                    break
                    
                self.message_queue.put({
                    "type": "progress", 
                    "value": i,
                    "status": f"Evaluating task {i+1}/20: {task_id}"
                })
                
                # Evaluate task
                task_start = time.time()
                try:
                    task = challenges[task_id]
                    result = solve_task(task)
                    
                    if 'attempt_1' in result:
                        prediction = result['attempt_1'][0]
                        gold_solution = solutions[task_id][0]
                        
                        is_correct, details, accuracy = grids_equal(prediction, gold_solution)
                        score = 1 if is_correct else 0
                        status = "PASS" if is_correct else "FAIL"
                        total_score += score
                        
                    else:
                        raise Exception("No attempt_1 in result")
                        
                except Exception as e:
                    score = 0
                    status = "ERROR"
                    accuracy = 0.0
                    details = str(e)
                
                duration = time.time() - task_start
                
                # Send result
                self.message_queue.put({
                    "type": "result",
                    "data": {
                        "task_id": task_id,
                        "status": status,
                        "score": score,
                        "accuracy": accuracy,
                        "duration": duration,
                        "details": details
                    }
                })
                
            # Send final summary
            total_time = time.time() - start_time
            success_rate = (total_score / len(task_ids)) * 100
            
            summary = f"""🎯 EVALUATION COMPLETE
            
Total Score: {total_score}/{len(task_ids)}
Success Rate: {success_rate:.1f}%
Total Time: {total_time:.1f}s
Average Time/Task: {total_time/len(task_ids):.1f}s

Performance Tier: {"🥇 GOLD" if success_rate >= 50 else "🥈 SILVER" if success_rate >= 25 else "🥉 BRONZE" if success_rate >= 10 else "📊 BASELINE"}
            """
            
            self.message_queue.put({"type": "summary", "text": summary})
            self.message_queue.put({"type": "complete"})
            
        except Exception as e:
            self.message_queue.put({
                "type": "summary", 
                "text": f"❌ Evaluation failed: {e}"
            })
            self.message_queue.put({"type": "complete"})

def main():
    """Main GUI function."""
    root = tk.Tk()
    app = ARCEvaluationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
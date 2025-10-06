"""
Fact generation module for the RFT-ILP neuro-symbolic architecture.

This module converts the output of the object extraction process into a
structured, relational format (logical facts) suitable for consumption by an
Inductive Logic Programming (ILP) engine.
"""

from typing import List, Dict, Any

def generate_facts(example_id: str, object_inventory: List[Dict[str, Any]], grid_type: str) -> List[str]:
    """
    Converts an object inventory into a list of Prolog-style logical facts.

    Args:
        example_id: A unique identifier for the task example (e.g., "ex1").
        object_inventory: A list of object dictionaries from connected_components.
        grid_type: "input" or "output" to distinguish the facts.

    Returns:
        A list of strings, where each string is a logical fact.
    """
    facts = []
    for i, obj in enumerate(object_inventory):
        obj_id = f"obj{i+1}"
        
        if grid_type == "input":
            facts.append(f"input_object({example_id}, {obj_id}).")
        elif grid_type == "output":
            facts.append(f"output_object({example_id}, {obj_id}).")

        # Add property facts
        color = obj.get("color")
        if color is not None:
            facts.append(f"has_property({example_id}, {obj_id}, color, {color}).")

        # The 'size' is the number of pixels in the object
        size = len(obj.get("pixels", []))
        if size > 0:
            facts.append(f"has_property({example_id}, {obj_id}, size, {size}).")

        # Placeholder for shape. This would require a shape recognition module.
        facts.append(f"has_property({example_id}, {obj_id}, shape, unknown).")

    facts.extend(_generate_relation_facts(example_id, object_inventory, grid_type))
    
    return facts

def _generate_relation_facts(example_id: str, object_inventory: List[Dict[str, Any]], grid_type: str) -> List[str]:
    """
    Generates relation facts between objects based on their bounding boxes.
    """
    facts = []
    for i, obj1 in enumerate(object_inventory):
        for j, obj2 in enumerate(object_inventory):
            if i == j:
                continue

            obj1_id = f"obj{i+1}"
            obj2_id = f"obj{j+1}"
            
            bbox1 = obj1.get("bbox")
            bbox2 = obj2.get("bbox")

            if not bbox1 or not bbox2:
                continue

            t1, l1, h1, w1 = bbox1
            t2, l2, h2, w2 = bbox2

            # is_above / is_below
            if t1 + h1 <= t2:
                facts.append(f"relation({example_id}, {grid_type}, {obj1_id}, {obj2_id}, is_above).")
                facts.append(f"relation({example_id}, {grid_type}, {obj2_id}, {obj1_id}, is_below).")

            # is_left_of / is_right_of
            if l1 + w1 <= l2:
                facts.append(f"relation({example_id}, {grid_type}, {obj1_id}, {obj2_id}, is_left_of).")
                facts.append(f"relation({example_id}, {grid_type}, {obj2_id}, {obj1_id}, is_right_of).")

            # is_enclosing / is_enclosed_by
            if t1 <= t2 and l1 <= l2 and (t1 + h1) >= (t2 + h2) and (l1 + w1) >= (l2 + w2):
                facts.append(f"relation({example_id}, {grid_type}, {obj1_id}, {obj2_id}, is_enclosing).")
                facts.append(f"relation({example_id}, {grid_type}, {obj2_id}, {obj1_id}, is_enclosed_by).")

            # is_touching
            # Simplified touching logic: bounding boxes are adjacent
            # Horizontal adjacency
            if (l1 + w1 == l2 or l2 + w2 == l1) and not (t1 + h1 <= t2 or t2 + h2 <= t1):
                 facts.append(f"relation({example_id}, {grid_type}, {obj1_id}, {obj2_id}, is_touching).")
            # Vertical adjacency
            elif (t1 + h1 == t2 or t2 + h2 == t1) and not (l1 + w1 <= l2 or l2 + w2 <= l1):
                facts.append(f"relation({example_id}, {grid_type}, {obj1_id}, {obj2_id}, is_touching).")

    return list(set(facts)) # Return unique facts

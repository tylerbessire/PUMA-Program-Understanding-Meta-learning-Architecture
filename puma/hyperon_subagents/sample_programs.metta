; ============================================================================
; PUMA MeTTa Sample Programs
; Demonstrates common operations for PUMA cognitive architecture
; ============================================================================

; ----------------------------------------------------------------------------
; 1. PATTERN MATCHING - Core operation for grid analysis
; ----------------------------------------------------------------------------

; Define some sample grid cells
(cell 0 0 blue)
(cell 1 0 blue)
(cell 2 0 red)
(cell 0 1 blue)
(cell 1 1 green)
(cell 2 1 blue)

; Query: Find all blue cells
!(match &self
    (cell ?x ?y blue)
    (cell ?x ?y blue))

; Query: Find cells in row 0
!(match &self
    (cell ?x 0 ?color)
    (cell ?x 0 ?color))

; ----------------------------------------------------------------------------
; 2. RELATIONAL FRAME THEORY - Analogical reasoning
; ----------------------------------------------------------------------------

; Define coordination frames (similarity relations)
(RelFrame coordination square rectangle 0.8)
(RelFrame coordination circle ellipse 0.7)
(RelFrame coordination triangle pyramid 0.6)

; Define opposition frames (contrasts)
(RelFrame opposition light dark 0.9)
(RelFrame opposition large small 0.9)
(RelFrame opposition hot cold 1.0)

; Define hierarchical frames (categorization)
(RelFrame hierarchy square category:shape 1.0)
(RelFrame hierarchy circle category:shape 1.0)
(RelFrame hierarchy triangle category:shape 1.0)
(RelFrame hierarchy blue category:color 1.0)
(RelFrame hierarchy red category:color 1.0)

; Query: Find shapes similar to square (coordination)
!(match &self
    (RelFrame coordination square ?similar ?strength)
    (RelFrame coordination square ?similar ?strength))

; Query: Find all opposites of "light"
!(match &self
    (RelFrame opposition light ?opposite ?strength)
    (RelFrame opposition light ?opposite ?strength))

; Query: Find all instances in the "shape" category
!(match &self
    (RelFrame hierarchy ?instance category:shape ?strength)
    (RelFrame hierarchy ?instance category:shape ?strength))

; ----------------------------------------------------------------------------
; 3. FREQUENCY LEDGER SYSTEM - PUMA's Core Innovation
; ----------------------------------------------------------------------------

; Define objects with frequency attributes
(object obj1 (frequency 3) (type pattern_a))
(object obj2 (frequency 3) (type pattern_a))
(object obj3 (frequency 1) (type pattern_b))
(object obj4 (frequency 3) (type pattern_a))
(object obj5 (frequency 2) (type pattern_c))

; Query: Group objects by frequency = 3 (high frequency pattern)
!(match &self
    (object ?id (frequency 3) ?type)
    (object ?id (frequency 3) ?type))

; Query: Find rare objects (frequency = 1)
!(match &self
    (object ?id (frequency 1) ?type)
    (object ?id (frequency 1) ?type))

; ----------------------------------------------------------------------------
; 4. TRANSFORMATION RULES - Pattern-based rewriting
; ----------------------------------------------------------------------------

; Define transformation rule: blue -> red
(transform-rule
    (pattern (cell ?x ?y blue))
    (replacement (cell ?x ?y red)))

; Define transformation rule: horizontal flip
(transform-rule
    (pattern (cell ?x ?y ?color))
    (replacement (cell (- max_x ?x) ?y ?color)))

; Define transformation rule: rotate 90 degrees
(transform-rule
    (pattern (cell ?x ?y ?color))
    (replacement (cell ?y (- max_y ?x) ?color)))

; ----------------------------------------------------------------------------
; 5. CAUSAL REASONING - Temporal and causal chains
; ----------------------------------------------------------------------------

; Define causal frames
(RelFrame causal action_a outcome_b 0.7)
(RelFrame causal outcome_b outcome_c 0.6)
(RelFrame causal action_x outcome_y 0.8)

; Define temporal frames
(RelFrame temporal event_1 event_2 1.0)
(RelFrame temporal event_2 event_3 1.0)
(RelFrame temporal event_3 event_4 1.0)

; Query: Find direct causal relationships
!(match &self
    (RelFrame causal ?cause ?effect ?strength)
    (RelFrame causal ?cause ?effect ?strength))

; Query: Find temporal sequences starting from event_1
!(match &self
    (RelFrame temporal event_1 ?next ?strength)
    (RelFrame temporal event_1 ?next ?strength))

; ----------------------------------------------------------------------------
; 6. DERIVATIONAL REASONING - Transitive relations
; ----------------------------------------------------------------------------

; If A is similar to B, and B is similar to C, then A is similar to C
; (with decaying strength)

(RelFrame coordination concept_a concept_b 0.9)
(RelFrame coordination concept_b concept_c 0.8)

; Derived frame (should be computed):
; (RelFrame coordination concept_a concept_c (* 0.9 0.8)) = 0.72

; Query to test derivation
!(match &self
    (RelFrame coordination concept_a ?target ?strength)
    (RelFrame coordination concept_a ?target ?strength))

; ----------------------------------------------------------------------------
; 7. COMPARATIVE REASONING - Magnitude and ordering
; ----------------------------------------------------------------------------

; Define comparative frames
(RelFrame comparative small medium 0.7)
(RelFrame comparative medium large 0.7)
(RelFrame comparative large huge 0.8)

; Define magnitude properties
(magnitude obj_a 5)
(magnitude obj_b 10)
(magnitude obj_c 3)
(magnitude obj_d 15)

; Query: Find objects with magnitude > 8
!(match &self
    (magnitude ?obj ?mag)
    (> ?mag 8))

; ----------------------------------------------------------------------------
; 8. SPATIAL REASONING - Location and proximity
; ----------------------------------------------------------------------------

; Define spatial frames
(RelFrame spatial cell_00 cell_01 1.0)  ; adjacent vertically
(RelFrame spatial cell_00 cell_10 1.0)  ; adjacent horizontally
(RelFrame spatial cell_11 cell_12 1.0)

; Define position predicates
(adjacent-horizontal (cell 0 0) (cell 1 0))
(adjacent-horizontal (cell 1 0) (cell 2 0))
(adjacent-vertical (cell 0 0) (cell 0 1))
(adjacent-vertical (cell 0 1) (cell 0 2))

; Query: Find all horizontal adjacencies
!(match &self
    (adjacent-horizontal ?cell1 ?cell2)
    (adjacent-horizontal ?cell1 ?cell2))

; ----------------------------------------------------------------------------
; 9. EPISODIC MEMORY - Experience tracking
; ----------------------------------------------------------------------------

; Define episodic memory nodes
(episode ep_001
    (timestamp "2025-01-15T10:30:00Z")
    (type exploration)
    (outcome success)
    (emotional_valence 0.8))

(episode ep_002
    (timestamp "2025-01-15T10:35:00Z")
    (type problem_solving)
    (outcome failure)
    (emotional_valence -0.3))

(episode ep_003
    (timestamp "2025-01-15T10:40:00Z")
    (type exploration)
    (outcome success)
    (emotional_valence 0.9))

; Query: Find successful episodes
!(match &self
    (episode ?id ?props)
    (= (get ?props outcome) success))

; Query: Find episodes with positive emotional valence
!(match &self
    (episode ?id (timestamp ?t) (type ?type) (outcome ?out) (emotional_valence ?v))
    (> ?v 0.5))

; ----------------------------------------------------------------------------
; 10. GOAL-DIRECTED REASONING - Intention and planning
; ----------------------------------------------------------------------------

; Define goals
(goal solve_arc_puzzle
    (priority high)
    (deadline "2025-12-31")
    (status active))

(goal learn_transformation_rules
    (priority medium)
    (deadline "2025-06-30")
    (status active))

(goal improve_pattern_recognition
    (priority high)
    (deadline "2025-03-31")
    (status completed))

; Query: Find active high-priority goals
!(match &self
    (goal ?name (priority high) ?other (status active))
    (goal ?name (priority high) ?other (status active)))

; ----------------------------------------------------------------------------
; 11. META-LEARNING - Learning to learn
; ----------------------------------------------------------------------------

; Track learning performance over time
(learning-trial trial_1 (accuracy 0.45) (epoch 1))
(learning-trial trial_2 (accuracy 0.52) (epoch 2))
(learning-trial trial_3 (accuracy 0.61) (epoch 3))
(learning-trial trial_4 (accuracy 0.68) (epoch 4))
(learning-trial trial_5 (accuracy 0.73) (epoch 5))

; Query: Find trials with accuracy > 0.6
!(match &self
    (learning-trial ?trial (accuracy ?acc) ?epoch)
    (> ?acc 0.6))

; ----------------------------------------------------------------------------
; 12. SELF-MODIFICATION - Code introspection
; ----------------------------------------------------------------------------

; Track code modifications
(code-version v1.0 (performance 0.65) (timestamp "2025-01-01"))
(code-version v1.1 (performance 0.71) (timestamp "2025-01-15"))
(code-version v1.2 (performance 0.68) (timestamp "2025-01-20"))

; Modification hypotheses
(modification-hypothesis hyp_1
    (description "Increase beam search width")
    (predicted_improvement 0.05)
    (confidence 0.7))

(modification-hypothesis hyp_2
    (description "Add frequency-based filtering")
    (predicted_improvement 0.08)
    (confidence 0.8))

; Query: Find best code version by performance
!(match &self
    (code-version ?version (performance ?perf) ?time)
    (> ?perf 0.70))

; ============================================================================
; End of PUMA MeTTa Sample Programs
; ============================================================================

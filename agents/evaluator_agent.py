import re
from state import DualReportState

def evaluator_node(state: DualReportState) -> dict:
    print("\n" + "=" * 60 + "\nSTEP 6 — Comparative Evaluator (Multi-Domain Heuristic Engine)\n" + "=" * 60)
    
    # Extract student text safely as a unified string
    raw_text = state.get("student_report", "")
    if isinstance(raw_text, list):
        text = " ".join([str(item) for item in raw_text]).lower()
    else:
        text = str(raw_text).lower()
        
    words = set(re.findall(r'\b\w+\b', text))
    
    # 1. Evidence & Quantitative Terms (Physics, CS, Math, Engineering, Data)
    evidence_words = {
        # Physics & Mechanics
        "formula", "equation", "law", "constant", "vector", "scalar", "mass", "force", "velocity", 
        "acceleration", "momentum", "energy", "joule", "newton", "pressure", "density", "ratio",
        "thrust", "impulse", "friction", "inertia", "gravity", "mass-flow", "equilibrium",
        # Computer Science & Data
        "data", "metric", "algorithm", "parameter", "variable", "dataset", "accuracy", "latency",
        "throughput", "complexity", "percentage", "rate", "value", "result", "sample", "benchmark",
        # General Academic & Empirical
        "evidence", "proof", "empirical", "quantitative", "measured", "observed", "calculated",
        "derived", "experiment", "analysis", "finding", "figure", "table", "magnitude"
    }
    
    # 2. Counterarguments, Contrast & Trade-offs
    counter_words = {
        "whereas", "conversely", "however", "although", "despite", "differ", "trade-off", "tradeoff",
        "limitation", "opposite", "resists", "reaction", "constraint", "drawback", "downside",
        "challenge", "bottleneck", "alternatively", "nonetheless", "contrary", "divergence",
        "exception", "risk", "disadvantage", "vulnerability", "compromise", "mitigate", "inverse",
        "inversely", "unbalances", "cancels", "cancellation"
    }
    
    # 3. Scope, Systems & Architecture
    scope_words = {
        "framework", "system", "boundary", "component", "isolation", "engineer", "model",
        "dynamics", "context", "architecture", "environment", "domain", "scope",
        "application", "infrastructure", "subsystem", "interaction", "mechanism", "macro", "micro",
        "scale", "lifecycle", "operational", "baseline", "perspective", "paradigm", "diagram",
        "reference", "body", "frame"
    }

    # 4. Logical Coherence & Structural Connectives
    coherence_words = {
        "therefore", "thus", "hence", "consequently", "because", "furthermore", "moreover",
        "illustrates", "demonstrates", "implies", "indicates", "synthesize", "integrating",
        "subsequently", "accordingly", "thereby", "signifies", "underlying", "establishes",
        "yields", "governs", "provides", "quantifies"
    }
    
    # Calculate scores (1 to 5 scale per dimension)
    evidence_score = min(5, max(1, len(words.intersection(evidence_words)) // 2 + 1))
    counter_score = min(5, max(1, len(words.intersection(counter_words)) * 2))
    scope_score = min(5, max(1, len(words.intersection(scope_words))))
    coherence_score = min(5, max(1, len(words.intersection(coherence_words)) + (2 if len(words) > 80 else 0)))
    
    total = coherence_score + evidence_score + counter_score + scope_score
    
    analysis = (
        f"### Paul-Elder Rubric Breakdown (Deterministic Score: {total}/20)\n"
        f"- **Logical Coherence**: {coherence_score}/5\n"
        f"- **Evidence Quality**: {evidence_score}/5\n"
        f"- **Counterargument Handling**: {counter_score}/5\n"
        f"- **Scope Awareness**: {scope_score}/5\n\n"
        f"**Evaluation Metrics Summary**:\n"
        f"- Unique words parsed: {len(words)}\n"
        f"- Quantitative/Evidence matches: {len(words.intersection(evidence_words))}\n"
        f"- Contrast/Trade-off matches: {len(words.intersection(counter_words))}\n"
        f"- System/Scope matches: {len(words.intersection(scope_words))}\n"
        f"- Logical transition matches: {len(words.intersection(coherence_words))}"
    )
    return {"comparative_analysis": analysis}
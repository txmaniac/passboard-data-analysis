import string
from typist import Typist, TemplateTypist
from policy import BlockingPolicy

class SimulationEngine:
    def __init__(self, risk_evaluator):
        self.risk_evaluator = risk_evaluator
        self.all_candidates = list(string.ascii_letters + string.digits + "!@#$%^&*")
        
    def run_trajectory(self, typist: Typist, policy: BlockingPolicy, target_length=12):
        password = ""
        blocked_events = 0
        risk_trace = []
        
        if isinstance(typist, TemplateTypist):
            typist.start_new_password()
            
        first_blocked_at = None
        
        for i in range(target_length):
            # 1. Determine Allowed Set
            allowed = policy.get_allowed_chars(password, self.all_candidates, self.risk_evaluator)
            
            # Heuristic for "Blocked": If allowed size < total size, blocking *was active*.
            if len(allowed) < len(self.all_candidates):
                blocked_events += 1
                if first_blocked_at is None:
                    first_blocked_at = i + 1 # 1-based index (1st char, 2nd char...)
            
            # 2. Ask Typist (Passing allowed constraints)
            char = typist.next_char(password, allowed_chars=allowed)
            
            if char is None:
                # Stuck or done
                break
                
            password += char
            
            # 3. Log Risk of the *actual* step (Post-intervention)
            # Re-eval risk for the chosen char
            step_risks = self.risk_evaluator(password[:-1], [char])
            risk_trace.append(step_risks.get(char, 0))
                
        return {
            "password": password,
            "length": len(password),
            "blocked_steps": blocked_events,
            "first_blocked_at": first_blocked_at,
            "risk_trace": risk_trace,
            "avg_step_risk": sum(risk_trace)/len(risk_trace) if risk_trace else 0
        }

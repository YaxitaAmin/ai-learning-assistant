# 4. src/data/generators/math_generator.py
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


class MathProblemGenerator:
    def __init__(self):
        self.problem_types = {
            'algebra': self._generate_algebra,
            'geometry': self._generate_geometry,
            'calculus': self._generate_calculus  # calculus method
        }

    def generate_problem(self, topic: str, difficulty: float) -> Dict:
        generator = self.problem_types.get(topic)
        if not generator:
            raise ValueError(f"Unknown topic: {topic}")
            
        problem, solution, explanation = generator(difficulty)
        
        return {
            'problem': problem,
            'solution': solution,
            'explanation': explanation,
            'difficulty': difficulty,
            'topic': topic
        }

    def _generate_algebra(self, difficulty: float) -> Tuple[str, float, str]:
        # Generate algebra problems with varying complexity
        if difficulty < 0.3:
            # Simple linear equations
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 20)
            solution = np.random.randint(1, 10)
            c = a * solution + b
            
            problem = f"Solve for x: {a}x + {b} = {c}"
            explanation = f"1. Subtract {b} from both sides\n" \
                          f"2. Divide both sides by {a}\n" \
                          f"3. x = {solution}"
                         
        elif difficulty < 0.7:
            # quadratic equations
            a = np.random.randint(1, 5)
            b = np.random.randint(-10, 10)
            c = np.random.randint(-10, 10)
            
            problem = f"Solve: {a}x² + {b}x + {c} = 0"
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                x1 = (-b + np.sqrt(discriminant)) / (2*a)
                x2 = (-b - np.sqrt(discriminant)) / (2*a)
                solution = sorted([x1, x2])
                explanation = "Using quadratic formula: x = (-b ± √(b² - 4ac)) / 2a"
            else:
                solution = "No real solutions"
                explanation = "Discriminant is negative, no real solutions exist"
        
        else:
            a1, a2 = np.random.randint(1, 5, size=2)
            b1, b2 = np.random.randint(1, 5, size=2)
            x, y = np.random.randint(-5, 5, size=2)
            c1 = a1*x + b1*y
            c2 = a2*x + b2*y
            
            problem = f"Solve the system:\n{a1}x + {b1}y = {c1}\n{a2}x + {b2}y = {c2}"
            solution = (x, y)
            explanation = "Solve using substitution or elimination method"
        
        return problem, solution, explanation

    def _generate_geometry(self, difficulty: float) -> Tuple[str, float, str]:
        # geometry solution
        fig, ax = plt.subplots(figsize=(6, 6))
        
        if difficulty < 0.3:
            # Tarea of triangle
            base = np.random.randint(3, 10)
            height = np.random.randint(3, 10)
           
            ax.plot([0, base, base/2, 0], [0, 0, height, 0], 'b-')
            ax.text(base/4, height/4, f"h={height}")
            ax.text(base/2, -0.5, f"b={base}")
            
            problem = "Find the area of the triangle"
            solution = (base * height) / 2
            explanation = f"Area = (base × height) / 2 = ({base} × {height}) / 2 = {solution}"
            
        else:
            radius = np.random.randint(3, 8)
            angle = np.random.randint(30, 150)
            
            circle = plt.Circle((0, 0), radius, fill=False)
            ax.add_artist(circle)
            ax.plot([0, radius*np.cos(np.radians(angle))], 
                    [0, radius*np.sin(np.radians(angle))], 'r-')
            
            problem = f"find the arc length for angle {angle}°"
            solution = 2 * np.pi * radius * (angle/360)
            explanation = f"arc length = 2πr × (θ/360°) = 2π×{radius}×({angle}/360)"
        
        # Convert plot to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        diagram = Image.open(buf)
        
        return problem, solution, explanation

    def _generate_calculus(self, difficulty: float) -> Tuple[str, float, str]:
        # generate calculus problems with varying complexity
        if difficulty < 0.5:
            # derivative problems
            problem = "Find the derivative of f(x) = 3x² + 5x - 7"
            solution = "f'(x) = 6x + 5"
            explanation = "The derivative of 3x² is 6x, and the derivative of 5x is 5."
        
        else:
            # integral problems
            problem = "Find the integral of f(x) = 3x² + 5"
            solution = "F(x) = x³ + 5x + C"
            explanation = "The integral of 3x² is x³, and the integral of 5 is 5x. Add the constant of integration C."
        
        return problem, solution, explanation

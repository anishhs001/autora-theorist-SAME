import pandas as pd
import itertools
import numpy as np

class ExpressionGenerator:
    '''
    This class generates all possible expressions from a list of items.
    columns: providing column headers to generate expressions from them
    binary_operators: list of binary operators to use
    unary_operators: list of unary operators to use
    '''
    def __init__(self, columns, max_expressions=10**5):
        # List of binary operators
        self.binary_operators = ['+', '*', '-', '/']

        # List of unary operators with their function calls
        self.unary_operators = [
            lambda x: f"np.exp({x})",
            lambda x: f"np.power({x}, 2)",
            lambda x: f"np.power({x}, 3)",
            lambda x: f"np.log({x}) if {x} > 0 else 0"
        ]

        # Initialize columns
        self.columns = columns
        # Max expressions threshold
        self.max_expressions = max_expressions
        # Initialize expression counter
        self.expression_count = 0

    def safe_power(self, base, exp):
        """Return a safe power expression to avoid invalid values."""
        if exp % 1 != 0:
            # For fractional exponents, ensure base is non-negative
            return f"np.power({base}, {exp}) if {base} >= 0 else 0"
        return f"np.power({base}, {exp})"

    def generate_unary_expressions(self, item):
        """Generate unary expressions for a single item."""
        unary_expressions = [item]
        for op in self.unary_operators:
            if self.expression_count >= self.max_expressions:
                return unary_expressions
            expression = op(item)
            unary_expressions.append(expression)
            self.expression_count += 1
        return unary_expressions

    def generate_polynomial_expressions(self, item):
        """Generate polynomial features by raising to different powers."""
        polynomial_expressions = []
        powers = [0.5, 1, 1.5, 2, 2.5, 3]
        for p in powers:
            if self.expression_count >= self.max_expressions:
                return polynomial_expressions
            expression = self.safe_power(item, p)
            polynomial_expressions.append(expression)
            self.expression_count += 1
        return polynomial_expressions

    def generate_combinations(self, items):
        """Generate combinations of unary and polynomial expressions."""
        combinations = []
        for item in items:
            unary_expressions = self.generate_unary_expressions(item)
            polynomial_expressions = self.generate_polynomial_expressions(item)
            all_expressions = unary_expressions + polynomial_expressions
            for i in range(len(all_expressions)):
                for j in range(i, len(all_expressions)):
                    if self.expression_count >= self.max_expressions:
                        return combinations
                    # Combine unary and polynomial expressions
                    combination = f"({all_expressions[i]}) + ({all_expressions[j]})"
                    combinations.append(combination)
                    self.expression_count += 1
                    combination = f"({all_expressions[i]}) * ({all_expressions[j]})"
                    combinations.append(combination)
                    self.expression_count += 1
        return combinations

    def generate_expressions(self, items):
        """Generate all possible expressions using binary and unary operators."""
        if self.expression_count >= self.max_expressions:
            return []
        if len(items) == 1:
            return (self.generate_unary_expressions(items[0]) +
                    self.generate_polynomial_expressions(items[0]))
        
        expressions = []
        for i in range(1, len(items)):
            left_parts = self.generate_expressions(items[:i])
            right_parts = self.generate_expressions(items[i:])
            for left in left_parts:
                for right in right_parts:
                    for op in self.binary_operators:
                        if self.expression_count >= self.max_expressions:
                            return expressions
                        if op == '/':
                            # Avoid division by zero
                            expression = f"({left}) {op} ({right})"
                            expressions.append(f"{expression} if ({right}) != 0 else 1")
                        else:
                            expressions.append(f"({left}) {op} ({right})")
                        self.expression_count += 1
        return expressions

    def generate_all_required_expressions(self):
        """Generate all expressions that use all columns."""
        all_expressions = set()
        combinations = itertools.permutations(self.columns)
        for combination in combinations:
            if self.expression_count >= self.max_expressions:
                break
            new_expressions = self.generate_expressions(list(combination))
            all_expressions.update(new_expressions)
            # Add polynomial and complex combinations
            complex_combinations = self.generate_combinations(list(combination))
            all_expressions.update(complex_combinations)
        return list(all_expressions)

    def dataframe_from_expr(self, df):
        """Generates all the new columns with the expressions mentioned in the dataframe"""
        expressions = self.generate_all_required_expressions()
        evaluated_columns = {}
        # Evaluate expressions and store the results in the dictionary
        for expr in expressions:
            try:
                # Use apply with a lambda function to evaluate each expression
                evaluated_columns[expr] = df.apply(lambda row: eval(expr, {'np': np}, row.to_dict()), axis=1)
            except Exception as e:
                print(f"Could not evaluate expression {expr}: {e}")
        return pd.DataFrame(evaluated_columns)

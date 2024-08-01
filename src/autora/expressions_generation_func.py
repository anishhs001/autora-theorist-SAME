import pandas as pd
import itertools
import numpy as np
import random

random.seed(42)

class ExpressionGenerator:
    '''
    This class generates all possible expressions from a list of items.
    columns: providing column headers to generate expressions from them
    binary_operators: list of binary operators to use
    unary_operators: list of unary operators to use
    '''
    def __init__(self, columns):
        # List of binary operators
        self.binary_operators = ['+', '*', '-', '/']

        # List of unary operators with their function calls
        self.unary_operators = [
            lambda x: f"np.exp({x})",
            lambda x: f"np.power({x}, 2)",
            lambda x: f"np.power({x}, 3)",
            lambda x: f"np.power({x}, 4)",
            lambda x: f"np.power({x}, 5)",
            lambda x: f"np.log({x})"
        ]
        
        # Initialize columns
        self.columns = columns

    def generate_unary_expressions(self, item):
        """Generate unary expressions for a single item."""
        unary_expressions = [item]
        for op in self.unary_operators:
            expression = op(item)
            # Ensure no invalid operations (e.g., log of non-positive values)
            if 'np.log' in expression:
                unary_expressions.append(expression + f" if {item} > 0 else 0")
            else:
                unary_expressions.append(expression)
        return unary_expressions

    def generate_expressions(self, items):
        """Generate all possible expressions using binary and unary operators."""
        if len(items) == 1:
            single_expressions = self.generate_unary_expressions(items[0])
            return single_expressions
        
        expressions = []
        for i in range(1, len(items)):
            left_parts = self.generate_expressions(items[:i])
            right_parts = self.generate_expressions(items[i:])
            for left in left_parts:
                for right in right_parts:
                    for op in self.binary_operators:
                        if op == '/':
                            # Avoid division by zero
                            expression = f"({left}) {op} ({right})"
                            expressions.append(f"{expression} if ({right}) != 0 else 1")
                        else:
                            expressions.append(f"({left}) {op} ({right})")
        return expressions

    def generate_all_expressions(self):
        """Generate all possible expressions for all non-empty subsets of columns."""
        all_expressions = set()
        # Generate all possible non-empty subsets of columns
        for i in range(1, len(self.columns) + 1):
            combinations = itertools.permutations(self.columns, i)
            for combination in combinations:
                all_expressions.update(self.generate_expressions(list(combination)))

        return list(all_expressions)
    
    def generate_all_required_expressions(self):
        """Filter expressions to ensure all columns are used."""
        all_expressions = self.generate_all_expressions()
        filtered_expressions = []

        for expr in all_expressions:
            # Check if all column names are in the expression
            if all(col in expr for col in self.columns):
                filtered_expressions.append(expr)
        
        return filtered_expressions
    
    def dataframe_from_expr(self, df):
        """Generates all the new columns with the expressions mentioned in the dataframe"""
        expressions = self.generate_all_required_expressions()
        if len(expressions) > 10**5:
            expressions = random.sample(expressions, 10**5)
        evaluated_columns = {}
        
        for expr in expressions:
            try:
                # Use apply with a lambda function to evaluate each expression
                evaluated_columns[expr] = df.apply(lambda row: eval(expr, {'np': np}, row.to_dict()), axis=1)
            except Exception as e:
                print(f"Could not evaluate expression {expr}: {e}")
        return pd.DataFrame(evaluated_columns)

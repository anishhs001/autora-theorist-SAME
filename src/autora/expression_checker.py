from sklearn.linear_model import Lasso
from expressions_generation_func import ExpressionGenerator
import pandas as pd


class ExpressionChecker:
    def __init__(self, df, y) -> None:
        self.df = df
        self.lasso = Lasso(alpha=0.1)
        self.y = y
        
    def check_coef(self, funct, max):
        coef_list = []

        for i, f in enumerate(funct):
            if f[i] < max:
                coef_list.append(f)

        return coef_list

    def sort_coef(self, f):

        if len(f) <= 1:
            return f
        pivot = f[len(f) // 2][len(f) // 2]
        left = [x for i, x in enumerate(f) if x[i] < pivot]
        middle = [x for i, x in enumerate(f) if x[i] == pivot]
        right = [x for i, x in enumerate(f) if x[i] > pivot]

        return self.sort_coef(left) + middle + self.sort_coef(right)
    
    def output(self):
        expression_generator = ExpressionGenerator(self.df.columns)
        evaluated_columns = expression_generator.evaluate_expressions(self.df)
        self.lasso.fit(self.df, self.y)
        coeficients = pd.Series(self.lasso.coef_, index=self.df.columns)
        coef_lists = [coeficients[i:i + 10] for i in range(0, len(coeficients), 10)]
        best_coef = []
        for eachone in coef_lists:
            sorted = self.sort_coef(eachone)
            topK = int(len(sorted)* 0.2)
            best_coef.append(self.check_coef(sorted, topK))

        return 



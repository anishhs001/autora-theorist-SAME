from expressions_generation_func import ExpressionGenerator
import pandas as pd
import numpy as np


class ExpressionChecker:
    def __init__(self, df, y):
        self.df = df
        self.y = y
    
    def output(self):
        expression_generator = ExpressionGenerator(self.df.columns)
        evaluated_columns = expression_generator.evaluate_expressions(self.df)
        # Create a new DataFrame from the evaluated columns
        result_df = pd.DataFrame(evaluated_columns)
        # Concatenate the new columns to the original DataFrame
        correlation_values = {}
        zero_variance_cols = result_df.columns[result_df.var() == 0]
        result_dataframe = result_df.drop(columns=zero_variance_cols)
        for column in result_dataframe.columns:
            correlation = np.corrcoef(result_dataframe[column], self.y)[0, 1]
            correlation_values[column] = np.abs(correlation)

        # Convert to a DataFrame for better readability
        correlation_df = pd.DataFrame(list(correlation_values.items()), columns=['Feature', 'Correlation'])
        correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)
        greatest_val = correlation_df.iloc[0]['Correlation']
        if len(correlation_df[correlation_df['Correlation']==greatest_val])>1:
            correlation_df['Length'] = correlation_df['Feature'].apply(len).reset_index(drop=True)
            # Find the index of the minimum length
            min_length_index = correlation_df['Length'].idxmin()
            feature_name = correlation_df['Feature'][min_length_index]
        else:
            feature_name = correlation_df.iloc[0]['Feature']

        return result_df



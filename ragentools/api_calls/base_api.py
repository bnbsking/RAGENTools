import os
import pandas as pd


class BaseAPI:
    def __init__(self, api_key: str, model_name: str, price_csv_path: str = "/app/ragentools/api_calls/prices.csv"):
        self.api_key = api_key
        self.model_name = model_name
        self.input_acc_tokens = 0
        self.output_acc_tokens = 0

        if os.path.exists(price_csv_path):
            print("Price CSV found, activate price computing")
            df = pd.read_csv(price_csv_path)
            df_row = df[df['model_name'] == model_name]
            self.input_token_price = df_row["input_token_price"].values[0] / 1e6
            self.output_token_price = df_row["output_token_price"].values[0] / 1e6
        else:
            self.input_token_price = 0.0
            self.output_token_price = 0.0

    def update_acc_tokens(self, input_tokens: int, output_tokens: int):
        self.input_acc_tokens += input_tokens
        self.output_acc_tokens += output_tokens

    def get_price(self) -> float:
        return self.input_acc_tokens * self.input_token_price + \
            self.output_acc_tokens * self.output_token_price
    
    def reset_acc_tokens(self):
        self.input_acc_tokens = 0
        self.output_acc_tokens = 0
    
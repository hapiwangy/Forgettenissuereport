import pandas as pd
class dreaddit_helper:
    def __init__(self, pm, TEMPLATE_STR):
        self.MAP_DREADDIT = {
            "1": "Stress", "0": "No Stress",
            1: "Stress", 0: "No Stress",
            "A": "Stress", "B": "No Stress",
            "a": "Stress", "b": "No Stress",
            "stress": "Stress", "no stress": "No Stress",
            "Stress": "Stress", "No Stress": "No Stress",
        }
        self.pm = pm
        self.TEMPLATE_STR = TEMPLATE_STR
    def load_df(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = ["subreddit", "text", "label"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"CSV {path} must contain columns: {required}")
        return df
    def normalize_label(self, x: str) -> str:
        x = str(x).strip()
        return self.MAP_DREADDIT.get(x, x)

    def build_example(self, row):
        # pm is the utils.prompts.prompt_maker
        # TEMPLATE_STR is the where template is        
        input_text = self.pm.dreaddit(self.TEMPLATE_STR, {"subreddit": row["subreddit"], "text": row["text"]})
        target = self.normalize_label(row["label"])
        full_text = f"{input_text} {target}".strip()
        return {"text": full_text}
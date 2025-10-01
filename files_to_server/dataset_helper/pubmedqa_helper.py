import pandas as pd
class pubmedqa_helper:
    def __init__(self, pm, TEMPLATE_STR):
        self.MAP_ABC = {
            "A": "Yes", "B": "No", "C": "Maybe",
            "a": "Yes", "b": "No", "c": "Maybe",
            "yes": "Yes", "no": "No", "maybe": "Maybe",
            "Yes": "Yes", "No": "No", "Maybe": "Maybe"
        }
        self.pm = pm
        self.TEMPLATE_STR = TEMPLATE_STR
    def load_df(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = ["question", "context", "final_decision"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"CSV {path} must contain columns: {required}")
        return df
    def normalize_label(self, x: str) -> str:
        x = str(x).strip()
        return self.MAP_ABC.get(x, x)

    def build_example(self, row):
        # pm is the utils.prompts.prompt_maker
        # TEMPLATE_STR is the where template is        
        input_text = self.pm.pubmedqa(self.TEMPLATE_STR, {"question": row["question"], "context": row["context"]})
        target = self.normalize_label(row["final_decision"])
        full_text = f"{input_text} {target}".strip()
        return {"text": full_text}

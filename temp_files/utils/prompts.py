class prompts_maker:
    def __init__(self):
        pass
    @staticmethod
    def PubMedQA(template, data_stream):
        return template.format(question=data_stream['question'], context=data_stream['context'])
    @staticmethod
    def dreaddit(template, data_stream):
        return template.format(subreddit=data_stream['subreddit'], text=data_stream['text'])
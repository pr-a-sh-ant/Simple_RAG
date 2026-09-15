class PromptService:
    def __init__(self):
        self.system = """
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.

            Answer:
        """
        self.template = """
            Context: {context}
            Question: {question}"""

        self.template_plain = """
            Question: {question}"""
        
        self.template = str(self.template)
        self.system = str(self.system)
        self.template_plain = str(self.template_plain)
    def get_prompt(self):
        return self.template, self.system

    def get_plain_prompt(self):
        return self.template_plain, self.system
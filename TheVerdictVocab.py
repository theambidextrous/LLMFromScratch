import re


class TheVerdictVocab:
    def __init__(self, filePath):
        self.file = filePath
        self.textFromFile = None

    def file_to_text(self):
        with open(self.file, "r", encoding="utf-8") as file:
            self.textFromFile = file.read()
    
    def vocabulary(self):
        self.file_to_text()
        preprosessed = re.split(r'([,.:;?_!"()\']|--|\s)', self.textFromFile)
        preprosessed = [word.strip() for word in preprosessed if word.strip()]
        unique_set = sorted(set(preprosessed))
        unique_set.extend(("<|unk|>", "<|endoftext|>"))
        vocabulary = { token:integer for integer,token in enumerate(unique_set) }
        return vocabulary

    def get_text(self):
        self.file_to_text()
        return self.textFromFile

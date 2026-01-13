import os
os.environ["SSL_CERT_FILE"] = "C:\\users\\e119897\\openaipublic.pem"

import requests
response = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe")
print(response.status_code)

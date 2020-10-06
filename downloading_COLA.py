import wget
import os
from zipfile import ZipFile

url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

if not os.path.exists('./cola_public_1.1.zip'):
    wget.download(url, './cola_public_1.1.zip')

with ZipFile('cola_public_1.1.zip', 'r') as zipObj:
    zipObj.extractall()




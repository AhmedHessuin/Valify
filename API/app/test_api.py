'''
this is for testing the api only 
'''
import requests

url = 'http://127.0.0.1:80/alive'
z=requests.post(url)
print(z.json())


url = 'http://127.0.0.1:80/inf'
files = {'uploaded_file': open('../input/image.txt', 'rb')}
z=requests.post(url, files=files)
print(z.json())

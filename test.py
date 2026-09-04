import urllib.request
try:
    print(urllib.request.urlopen("http://localhost:8000/app.js").getcode())
except Exception as e:
    print(e)

import requests
import numpy as np
import time

# get请求
while True:
    r = requests.get("https://voteapi.btime.com/v1/poll/voting?callback=jQuery1113015244814967428333_1565195500396&userId=2790965&userToken=e21d6f4f2775393af2ae76cb8bb61124&depMd5=889675bcbdc82c50260173446d19d4bd&subjId=2672&id=12103&_=1565195500421")
    r.set
    print(r.status_code)
    print(r.text)
    time.sleep(2)
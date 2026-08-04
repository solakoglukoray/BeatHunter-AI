"""
The task_id the API hands out must be fetchable from /task_status/.

It was not. task_id was f"yt_{query.lower()}" — the raw user input — so pasting a
YouTube URL put slashes in it. A FastAPI path parameter matches a single path segment,
and Starlette decodes %2F back to / before routing, so the client cannot escape its way
out: every poll 404'd and the UI sat there until its 180-second timeout and reported
"Analysis took too long". Nothing was slow. Nothing was even wrong with the analysis.

Search terms have no slashes, which is why "Travis Scott Type Beat" worked and a URL did
not, and why nobody noticed. The upload path was never affected because its key is a
hash.

    python test_task_id_routing.py          # needs the API on 127.0.0.1:8000
"""

import sys
import urllib.parse

import requests

API = "http://127.0.0.1:8000"
USER, PASSWORD, EMAIL = "routing_probe", "probe1234", "probe@example.com"

# Three, not more: the API rate-limits non-admin users to 3 analyses per minute, and a
# 429 here would look like a routing failure when it is the limiter doing its job.
QUERIES = [
    ("https://www.youtube.com/watch?v=h7NBEcRgBos", "tam URL"),
    ("https://youtu.be/h7NBEcRgBos",                "kisa URL"),
    ("Metro Boomin / dark type beat",               "slash iceren terim"),
]


def token():
    requests.post(f"{API}/register", json={"username": USER, "email": EMAIL,
                                           "password": PASSWORD}, timeout=10)
    r = requests.post(f"{API}/login", data={"username": USER, "password": PASSWORD},
                      timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def main():
    try:
        head = {"Authorization": f"Bearer {token()}"}
    except Exception as e:
        print(f"API'ye ulasilamadi ({e}) — once uvicorn'u baslat")
        return 2

    ok = True
    print(f"{'sonuc':6s} {'HTTP':>5s}  {'girdi':32s} task_id")
    for query, note in QUERIES:
        r = requests.post(f"{API}/analyze_youtube/", data={"query": query},
                          headers=head, timeout=30)
        if r.status_code != 200:
            print(f"  !!   {r.status_code}   {note:32s} (analyze_youtube reddetti)")
            ok = False
            continue
        task_id = r.json()["task_id"]

        # The client encodes exactly as app.py does.
        enc = urllib.parse.quote(task_id, safe="")
        s = requests.get(f"{API}/task_status/{enc}", headers=head, timeout=15)
        good = s.status_code == 200
        ok &= good
        print(f"  {'PASS' if good else 'FAIL':4s} {s.status_code:5d}  {note:32s} {task_id[:46]}")

    print("\n" + ("TUM TASK_ID LER POLLANABILIR" if ok else
                  "!!! POLLANAMAYAN task_id VAR — UI bunlarda timeout'a duser"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

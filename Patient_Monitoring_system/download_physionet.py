"""
download_physionet.py
---------------------
Downloads 1,000 PhysioNet/CinC 2019 Challenge Training Set A .psv files
directly from the public AWS S3 mirror (fast, open, reliable, no rate-limiting).
"""

import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join("data", "physionet2019")
BASE_URL = "https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/training_setA/"

TARGET_COUNT = 1000
THREADS = 16

os.makedirs(DATA_DIR, exist_ok=True)


def download_one(pid_num):
    fname = f"p{pid_num:06d}.psv"
    dest = os.path.join(DATA_DIR, fname)

    if os.path.exists(dest) and os.path.getsize(dest) > 50:
        return (fname, "skip")

    url = BASE_URL + fname
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            if len(data) < 50:
                return (fname, "empty")
            with open(dest, "wb") as f:
                f.write(data)
            return (fname, "ok")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return (fname, "404")
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return (fname, "error")


def main():
    existing = [f for f in os.listdir(DATA_DIR) if f.endswith(".psv")]
    print(f"[download] PhysioNet 2019 S3 Fast Downloader")
    print(f"[download] Target count: {TARGET_COUNT} files | Existing: {len(existing)} files")
    print(f"[download] Mirror: {BASE_URL}")

    all_ids = list(range(1, TARGET_COUNT + 1))
    stats = {"ok": 0, "skip": 0, "404": 0, "error": 0}
    start = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(download_one, pid): pid for pid in all_ids}
        for future in as_completed(futures):
            fname, status = future.result()
            done += 1
            stats[status] = stats.get(status, 0) + 1

            if done % 100 == 0 or done == len(all_ids):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(all_ids) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(all_ids)}] Downloaded: {stats['ok']} | Skipped: {stats['skip']} | Errors: {stats['error']} | {rate:.1f} files/s | ETA: {eta:.0f}s")

    total = len([f for f in os.listdir(DATA_DIR) if f.endswith(".psv")])
    elapsed = time.time() - start
    print(f"\n[download] DONE in {elapsed:.1f}s")
    print(f"[download] data/physionet2019/ now has {total} .psv files")


if __name__ == "__main__":
    main()

"""
Frugal AAPL price extraction from Zihan1004/FNSPID.

The repo exposes per-ticker OHLC only inside Stock_price/full_history.zip (589 MB).
We must NOT download the whole archive. Instead we expose the remote zip as a
seekable file via HTTP Range requests and let the stdlib `zipfile` read only the
central directory + the single AAPL member (a few hundred KB transferred).
"""
import io
import os
import zipfile
import urllib.request

ZIP_URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/"
           "resolve/main/Stock_price/full_history.zip")
DATA_DIR = "./real_world_fnspid/data"
OUT_PATH = os.path.join(DATA_DIR, "AAPL_prices.csv")


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file backed by HTTP Range requests."""

    def __init__(self, url):
        self.url = url
        self._pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.accept_ranges = r.headers.get("Accept-Ranges", "none")
        self._bytes_fetched = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        if n == 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + n, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self._pos += len(data)
        self._bytes_fetched += len(data)
        return data


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Opening remote zip via HTTP Range (no full download)...")
    remote = HttpRangeFile(ZIP_URL)
    print(f"  archive size: {remote.size/1e6:.1f} MB | Accept-Ranges: {remote.accept_ranges}")
    if remote.accept_ranges.lower() != "bytes":
        raise SystemExit("Server does not advertise byte-range support; aborting.")

    zf = zipfile.ZipFile(remote)
    names = zf.namelist()
    # locate the AAPL member (e.g. 'full_history/AAPL.csv')
    cand = [n for n in names if os.path.basename(n).upper() == "AAPL.CSV"]
    print(f"  members in archive: {len(names)} | AAPL match: {cand}")
    if not cand:
        sample = [n for n in names[:10]]
        raise SystemExit(f"No AAPL.csv inside archive. First members: {sample}")

    member = cand[0]
    raw = zf.read(member)
    with open(OUT_PATH, "wb") as f:
        f.write(raw)
    print(f"Extracted '{member}' -> {OUT_PATH} ({len(raw)/1e6:.2f} MB on disk)")
    print(f"Total bytes transferred over network: {remote._bytes_fetched/1e6:.2f} MB "
          f"(vs {remote.size/1e6:.0f} MB full archive)")


if __name__ == "__main__":
    main()

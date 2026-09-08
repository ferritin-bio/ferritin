#!/usr/bin/env python3
"""List a remote checkpoint's tensors without downloading it.

Every real finding in the recent PLM ports came from opening the checkpoint
rather than from reading the Rust: ProtT5's container turned out to have no
safetensors and a 28-piece SentencePiece vocabulary behind a declared
`vocab_size` of 128 (ferritin-goh.5); ESM3's structure encoder had seven
defects; ESMFold2 turned out to be an AlphaFold3-class model. Downloading
multi-gigabyte weights to learn a tensor name is slow enough that it does not
happen, so this reads just the index:

* **safetensors** — the header is a length-prefixed JSON blob at offset 0, so
  one range request of a few KB names every tensor.
* **torch `.bin` / `.pth`** — a zip. Read the end-of-central-directory record,
  then the central directory, then just the `data.pkl` member, and unpickle it
  with stubs in place of torch's classes. Tensor storages are *not* read.

Enumerating a 2.42 GB ProtT5 checkpoint this way costs ~60 KB and 7 requests.

Usage
-----
    # a repo's file list and config.json
    python scripts/peek_checkpoint.py Rostlab/prot_t5_xl_half_uniref50-enc

    # tensor names, shapes and dtypes from one weight file
    python scripts/peek_checkpoint.py Rostlab/prot_t5_xl_half_uniref50-enc \
        pytorch_model.bin

The pickle is never executed: `find_class` returns inert stubs, so a malicious
checkpoint cannot run code through this path. It still comes from the network,
so treat the *names* it prints as untrusted text.
"""
import io, json, pickle, struct, sys, urllib.request, zipfile

class R:
    """Minimal seekable file-like over HTTP range requests."""
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            # HF redirects to CDN; keep the resolved url
            self.url = r.geturl()
        self.pos = 0
        self.reads = 0
        self.bytes = 0
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def tell(self):
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        req = urllib.request.Request(self.url)
        req.add_header("Range", f"bytes={self.pos}-{end}")
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.reads += 1
        self.bytes += len(data)
        self.pos += len(data)
        return data
    def readable(self): return True
    def seekable(self): return True

class ODict(dict):
    """dict that tolerates pickle's BUILD/__setstate__ on an OrderedDict."""
    def __setstate__(self, state): pass

class Stub:
    def __init__(self, name): self.name = name
    def __call__(self, *a, **k): return ("CALL", self.name, a)
    def __setstate__(self, state): pass
    def __repr__(self): return f"<{self.name}>"

def rebuild(storage, storage_offset, size, stride, *rest):
    # storage is ("STORAGE", dtype, key, device, numel)
    dtype = storage[1] if isinstance(storage, tuple) else "?"
    return {"shape": list(size), "dtype": dtype}


def rebuild_from_type_v2(func, new_type, args, state):
    """Unwrap `torch._tensor._rebuild_from_type_v2`.

    Checkpoints saved from a tensor subclass (ESM-C's are) wrap every entry in
    this, so without unwrapping every shape prints as the raw pickle call.
    """
    return func(*args) if callable(func) else {"shape": None, "dtype": "?"}

class Unp(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "_rebuild_tensor_v2":
            return rebuild
        if name == "_rebuild_from_type_v2":
            return rebuild_from_type_v2
        if name == "OrderedDict":
            return ODict
        return Stub(f"{mod}.{name}")
    def persistent_load(self, pid):
        # ('storage', <dtype stub>, key, device, numel)
        if isinstance(pid, tuple) and len(pid) >= 5:
            d = pid[1]
            dn = getattr(d, "name", str(d)).split(".")[-1].replace("Storage", "")
            return ("STORAGE", dn, pid[2], pid[3], pid[4])
        return ("STORAGE", "?", None, None, None)

def safetensors_header(url: str) -> dict:
    """The tensor index of a remote safetensors file: 8-byte LE length, then JSON."""
    f = R(url)
    f.seek(0)
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    print(f"# {f.size / 1e9:.2f} GB file; header is {n} bytes, "
          f"read in {f.reads} ranges / {f.bytes / 1e6:.2f} MB\n")
    return header


def torch_pickle_tensors(url: str) -> dict:
    """The state dict of a remote torch zip-pickle, storages not read."""
    f = R(url)
    z = zipfile.ZipFile(f)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")]
    if not pkl:
        raise SystemExit(f"no data.pkl among {len(z.namelist())} zip entries")
    print(f"# {len(z.namelist())} zip entries, {f.size / 1e9:.2f} GB total; "
          f"reading {pkl[0]}")
    obj = Unp(io.BytesIO(z.read(pkl[0]))).load()
    print(f"# read {f.reads} ranges / {f.bytes / 1e6:.2f} MB\n")
    return obj


def hf_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def show_repo(repo: str) -> None:
    """The repo's file list and config.json — what a port needs before starting."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}") as r:
        meta = json.load(r)
    print("=== files ===")
    for sibling in meta.get("siblings", []):
        print(" ", sibling["rfilename"])
    print("=== config.json ===")
    try:
        with urllib.request.urlopen(hf_url(repo, "config.json")) as r:
            print(json.dumps(json.load(r), indent=1))
    except Exception as e:  # noqa: BLE001 - a missing config is informative, not fatal
        print("(none:", e, ")")


def show_tensors(repo: str, filename: str, revision: str = "main") -> None:
    url = hf_url(repo, filename, revision)
    if filename.endswith(".safetensors"):
        header = safetensors_header(url)
        entries = [
            (k, v.get("shape"), v.get("dtype"))
            for k, v in header.items()
            if k != "__metadata__"
        ]
    else:
        obj = torch_pickle_tensors(url)
        if not isinstance(obj, dict):
            print("top-level object is", type(obj).__name__, "not a state dict:", obj)
            return
        entries = [
            (k, v["shape"], v["dtype"]) if isinstance(v, dict) else (k, None, repr(v))
            for k, v in obj.items()
        ]

    print(f"{len(entries)} tensors\n")
    for name, shape, dtype in entries:
        print(f"{name:70s} {str(shape):22s} {dtype}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    repo = sys.argv[1]
    if len(sys.argv) == 2:
        show_repo(repo)
    else:
        show_tensors(repo, sys.argv[2], *sys.argv[3:4])

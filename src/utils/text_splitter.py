import re


class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = [
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]  # paragraph → line → sentence → word → char

    def _split(self, text, separators, size):
        # break text into pieces <= size, using the largest separator that works
        if len(text) <= size:
            return [text]
        sep, rest = separators[0], separators[1:]
        if not sep:
            return [text[i : i + size] for i in range(0, len(text), size)]
        parts = text.split(sep)
        # keep the separator on each piece so nothing is lost when pieces are rejoined
        pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
        out = []
        for p in pieces:
            out.extend(self._split(p, rest, size))
        return out

    def _merge(self, pieces, size):
        # pack neighbouring pieces into chunks, across all split levels
        chunks, buf = [], ""
        for p in pieces:
            if buf and len(buf) + len(p) > size:
                chunks.append(buf)
                buf = ""
            buf += p
        if buf:
            chunks.append(buf)
        return [c.strip() for c in chunks if c.strip()]

    def split_text(self, text, metadata=None):
        metadata = metadata or {}
        # reserve room for the overlap and its joining space so chunks never exceed chunk_size
        size = self.chunk_size - (self.chunk_overlap + 1 if self.chunk_overlap else 0)
        raw = self._merge(self._split(text, self.separators, size), size)
        chunks = []
        for i, c in enumerate(raw):
            if i > 0 and self.chunk_overlap:
                prev = raw[i - 1]
                tail = prev[-self.chunk_overlap :]
                if len(prev) > self.chunk_overlap:
                    tail = re.sub(r"^\S*\s*", "", tail)  # drop the first word, it may be cut
                if tail:
                    c = tail + " " + c
            chunks.append({"content": c, "metadata": dict(metadata)})
        return chunks


if __name__ == "__main__":
    text = (
        "Printers jam often. Open the tray to fix it.\n"
        "Then restart.\n\n"
        "Error err-4021 means paper jam. Error err-4022 means low ink. Replace the cartridge soon.\n\n"
        "Warranty is two years."
    )
    plain = [c["content"] for c in TextSplitter(40, 0).split_text(text)]
    assert all(len(c) <= 40 for c in plain)
    assert plain[0] == "Printers jam often."  # sentence period kept
    assert "Open the tray to fix it.\nThen restart." in plain  # small pieces merged across levels
    # every word survives splitting
    assert " ".join(plain).split() == text.split()

    overlapped = [c["content"] for c in TextSplitter(40, 10).split_text(text)]
    assert all(len(c) <= 40 for c in overlapped)
    words = set(text.split())
    for c in overlapped:
        assert set(c.split()) <= words, c  # no glued or half words

    blob = [c["content"] for c in TextSplitter(10, 0).split_text("a" * 25 + " bb")]
    assert all(len(c) <= 10 for c in blob), blob  # long words get cut by characters

    print("ok")
    for c in overlapped:
        print(len(c), repr(c))

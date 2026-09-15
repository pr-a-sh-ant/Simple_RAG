class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]   # paragraph → line → sentence → word → char

    def _split(self, text, separators):
        if len(text) <= self.chunk_size or not separators:
            return [text]
        sep, rest = separators[0], separators[1:]
        piece = text.split(sep) if sep else [text]
        out, buf = [], ""
        for p in piece:
            candidate = (buf + sep + p) if buf else p
            if len(candidate) <= self.chunk_size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                if len(p) > self.chunk_size:
                    out.extend(self._split(p, rest))
                    buf = ""
                else:
                    buf = p
                    
        if buf:
            out.append(buf)
        return out

    def split_text(self, text, metadata=None):
        metadata = metadata or {}
        raw = self._split(text, list(self.separators))
        chunks = []
        for i, c in enumerate(raw):
            if i > 0 and self.chunk_overlap:
                c = raw[i-1][-self.chunk_overlap:] + c
            chunks.append({"content": c, "metadata": dict(metadata)})
        return chunks
                

        
    
            
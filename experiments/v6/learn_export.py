"""Rebuild the winner-weighted policy from verified yan-v5 and the four fitted tensors."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import re

import numpy as np


HEAD_KEYS = ('hw.w', 'hw.b', 'hb.w', 'hb.b')


def export(base, heads, output):
    source = base.read_bytes()
    with np.load(heads, allow_pickle=False) as learned:
        if hashlib.sha256(source).hexdigest() != str(learned['base_sha256']):
            raise ValueError('Expected the frozen yan-v5 source specified in the fitted heads')
        text = source.decode()
        pattern = r'^_WEIGHTS = "([^"]+)"$'
        match = re.search(pattern, text, re.M)
        with np.load(io.BytesIO(base64.b64decode(match[1])), allow_pickle=False) as original:
            parameters = dict(original)
        for key in HEAD_KEYS:
            if parameters[key].shape != learned[key].shape:
                raise ValueError(f'Unexpected shape for {key}')
            parameters[key] = learned[key]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **parameters)
    payload = base64.b64encode(buffer.getvalue()).decode()
    content = text[:match.start(1)] + payload + text[match.end(1):]
    compile(content, str(output), 'exec')
    output.write_text(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('heads', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.base, args.heads, args.output)

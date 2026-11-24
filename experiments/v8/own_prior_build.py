"""Replace the v8 own-search prior slots with a refitted model; keep everything else."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCE_SHA = 'a87f8e6b83656c771cb5a25d6b6f383b245127e7c782e4c6c7c4a0a2cc15d396'


def main(a):
    source = a.source.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError('source is not the frozen v8 file')
    model = json.loads(a.model.read_text())
    coef = f"_V7_OWN_COEF=np.asarray({model['coefficients']!r},np.float32)"
    scale = f"_V7_OWN_SCALE=np.asarray({model['scale']!r},np.float32)"
    text, n_coef = re.subn(r'^_V7_OWN_COEF=.*$', coef, source, count=1, flags=re.M)
    text, n_scale = re.subn(r'^_V7_OWN_SCALE=.*$', scale, text, count=1, flags=re.M)
    if (n_coef, n_scale) != (1, 1):
        raise ValueError(f'expected one prior slot each, replaced {n_coef}/{n_scale}')
    compile(text, str(a.output), 'exec')
    if len(text.encode()) >= 10_000_000:
        raise ValueError('candidate exceeds the 10 MB tournament limit')
    a.output.write_text(text)
    report = dict(source_sha256=SOURCE_SHA, model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                  output_bytes=a.output.stat().st_size, coefficients=model['coefficients'],
                  scale=model['scale'], slots_replaced=n_coef + n_scale)
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('built', a.output, report['output_bytes'], report['output_sha256'][:12], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('model', type=Path)
    p.add_argument('output', type=Path)
    main(p.parse_args())

"""Compare CPU population plus mapped-byte consumption on one private Linux file.

This is a component benchmark, not model inference. Never run alongside a timed
model comparison. Cache eviction advice targets only this benchmark's own file.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


FLAG = 'FREETOKEN_PREFILL_POPULATE_SKIP_RESIDENT'


def consume(view, rows, row_bytes):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(view[row * row_bytes:(row + 1) * row_bytes])
    return digest.hexdigest()


def io_counts():
    return {key: int(value) for key, value in
            (line.split(': ', 1) for line in Path('/proc/self/io').read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True,
                        help='Existing disk directory for a private temporary file')
    parser.add_argument('--rows', type=int, default=128)
    parser.add_argument('--row-kib', type=int, default=2048)
    parser.add_argument('--scratch-mib', type=int, default=32)
    parser.add_argument('--pairs', type=int, default=4)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('requires Linux page-cache advice and mincore')
    if (args.rows < 4 or args.rows % 4 or args.pairs < 2 or args.pairs % 2
            or min(args.row_kib, args.scratch_mib) <= 0):
        parser.error('rows must be a positive multiple of four; pairs must be positive and even')
    if args.rows * args.row_kib > 2 * 1024 * 1024 or args.scratch_mib > 64:
        parser.error('component test is bounded to a 2 GiB file and 64 MiB scratch')
    if not args.directory.is_dir():
        parser.error('directory must already exist')

    import torch
    from freetoken.moe import host_banks
    from freetoken.moe import resident_range
    from freetoken.moe.resident_range import ResidentRangeProbe

    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), repo / 'python/freetoken/moe/host_banks.py',
               repo / 'python/freetoken/moe/resident_range.py']
    assert Path(host_banks.__file__).resolve() == sources[1], 'wrong host-bank module loaded'
    assert Path(resident_range.__file__).resolve() == sources[2], 'wrong residency module loaded'
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    row_bytes = args.row_kib * 1024
    rows = list(range(0, args.rows, 2))
    scratch = bytearray(args.scratch_mib << 20)
    probe = ResidentRangeProbe(len(scratch))
    samples = []
    prior_flag = os.environ.get(FLAG)
    try:
        with tempfile.TemporaryDirectory(prefix='freetoken-populate-', dir=args.directory) as directory:
            path = Path(directory) / 'private.ftw'
            expected = hashlib.sha256()
            with path.open('wb') as stream:
                stream.write(b'prefix-offset-017')
                for row in range(args.rows):
                    # Every row has different bytes, generated with bounded memory.
                    block = hashlib.sha256(str(row).encode()).digest() * (row_bytes // 32)
                    stream.write(block)
                    if row in rows:
                        expected.update(block)
                stream.flush()
                os.fsync(stream.fileno())
            assert path.stat().st_size == 17 + args.rows * row_bytes
            with host_banks.requested_hugepages('off'):
                bank = host_banks.HostBank((args.rows, row_bytes), torch.uint8,
                                          backing='file', file_path=str(path), file_offset=17)
            view = bank.memoryview()
            fd = os.open(path, os.O_RDONLY)
            try:
                def prepare(case):
                    # Drop this mapping's PTEs and this private inode's cached pages.
                    host_banks._madvise(bank._mapping_addr, bank._mapping_length, mmap.MADV_DONTNEED)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    warm = rows if case == 'warm' else rows[:len(rows) // 2] if case == 'mixed' else []
                    if warm:
                        consume(view, warm, row_bytes)
                    resident = sum(probe.resident(bank.addr + row * row_bytes, row_bytes) for row in rows)
                    if resident != len(warm):
                        raise RuntimeError(f'{case}: wanted {len(warm)} resident rows, observed {resident}')
                    return resident

                for case in ('warm', 'cold', 'mixed'):
                    # One warmup per mode exercises both paths in each case.
                    schedule = [(True, 0, mode) for mode in ('off', 'on')]
                    schedule += [(False, pair + 1, mode) for pair in range(args.pairs)
                                 for mode in (('off', 'on') if pair % 2 == 0 else ('on', 'off'))]
                    for warmup, pair, mode in schedule:
                        resident = prepare(case)
                        os.environ[FLAG] = '1' if mode == 'on' else '0'
                        before = io_counts()
                        begin = time.perf_counter()
                        copied = bank.populate_rows(rows, scratch)
                        populated = time.perf_counter()
                        actual = consume(view, rows, row_bytes)
                        ended = time.perf_counter()
                        after = io_counts()
                        assert actual == expected.hexdigest(), 'mapped bytes differ after population'
                        sample = dict(case=case, warmup=warmup, pair=pair, mode=mode,
                                      resident_rows_before=resident, selected_rows=len(rows),
                                      selected_bytes=len(rows) * row_bytes, scratch_bytes_read=copied,
                                      populate_s=populated - begin, consume_s=ended - populated,
                                      total_s=ended - begin, checksum=actual,
                                      io_delta={key: after[key] - before[key] for key in before})
                        samples.append(sample)
            finally:
                os.close(fd)
                view.release()
    finally:
        if prior_flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = prior_flag

    grouped = defaultdict(dict)
    for case in ('warm', 'cold', 'mixed'):
        for mode in ('off', 'on'):
            matching = [s for s in samples if s['case'] == case and s['mode'] == mode and not s['warmup']]
            grouped[case][mode] = dict(total_s=sum(s['total_s'] for s in matching),
                                      median_total_s=statistics.median(s['total_s'] for s in matching),
                                      scratch_bytes_read=sum(s['scratch_bytes_read'] for s in matching),
                                      physical_read_bytes=sum(s['io_delta']['read_bytes'] for s in matching))
        grouped[case]['wall_reduction_percent'] = 100 * (
            1 - grouped[case]['on']['total_s'] / grouped[case]['off']['total_s'])
    assert hashes == {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert revision == subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    print(json.dumps(dict(component_only=True, model_throughput_verified=False,
                          platform=platform.platform(), revision=revision,
                          file_bytes=17 + args.rows * row_bytes, row_bytes=row_bytes,
                          scratch_bytes=len(scratch), sources=hashes, hugepage_policy='off',
                          cache_preparation='Private-file DONTNEED, then selected-row reads for warm/mixed. '
                                            'Fully resident selected-row count is checked before each sample.',
                          cases=grouped, samples=samples), indent=2))


if __name__ == '__main__':
    main()

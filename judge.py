"""Generate semantic and extra-event adjudications, then compute DCIST."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from judge_client import ChatClient, ReviewError, load_config
from judge_pipeline import prepare, run_pipeline, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('prepare', help='Snapshot inputs and prompts without API calls')
    make.add_argument('--manifest', type=Path, required=True)
    make.add_argument('--run-dir', type=Path, required=True)
    run = sub.add_parser('run', help='Send review requests using your configured service')
    run.add_argument('--run-dir', type=Path, required=True)
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--max-requests', type=int, default=100)
    run.add_argument('--retry-failed', action='store_true', help='Retry each failed/unknown attempt at most once this run')
    args = parser.parse_args()
    lock = None
    try:
        require(args.run_dir.exists() or args.command == 'prepare', 'Run directory does not exist')
        args.run_dir.mkdir(parents=True, exist_ok=True)
        lock_path = args.run_dir / 'run.lock'
        try:
            lock = lock_path.open('x', encoding='utf-8')
        except FileExistsError:
            raise ReviewError('Run is locked; do not start duplicate requests. If the process crashed, '
                              'confirm it has stopped before removing run.lock.') from None
        lock.write(str(os.getpid()))
        lock.flush()
        if args.command == 'prepare':
            output = prepare(args.manifest, args.run_dir)
        else:
            require(args.workers > 0 and args.max_requests > 0, 'Workers and request limit must be positive')
            client = ChatClient(load_config(args.config))
            output = run_pipeline(args.run_dir, client, args.workers, args.max_requests, args.retry_failed)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0 if output.get('summary', {}).get('status', 'complete') == 'complete' else 2
    except ReviewError as exc:
        print(json.dumps({'status': 'review_error', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (ValueError, TypeError, KeyError, AssertionError, OSError, RuntimeError):
        print(json.dumps({'status': 'input_or_review_error',
                          'error': 'Input, artifact or response validation failed; inspect the run records.'}), file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            lock.close()
            lock_path.unlink()


if __name__ == '__main__':
    sys.exit(main())

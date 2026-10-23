"""Command-line interface for offline DCIST scoring."""
import argparse
import json
from pathlib import Path
import sys
from .scoring import compute

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('raw', 'reviewed'), required=True)
    parser.add_argument('--gold', type=Path)
    parser.add_argument('--prediction', type=Path)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--time', choices=('on', 'off'))
    parser.add_argument('--inference-status', choices=('completed', 'no_answer'))
    parser.add_argument('--case', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--no-answer-policy', choices=('error', 'zero'), default='error')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = compute(args)
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(rendered)
        else:
            print(rendered, end='')
        return 0 if result['summary']['status'] == 'complete' else 2
    except (ValueError, KeyError, TypeError, IndexError, AssertionError, OSError) as exc:
        print(json.dumps({'status': 'input_or_review_error', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

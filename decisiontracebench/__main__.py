"""Run the scoring or automated-adjudication command."""
import sys
from . import cli_score, cli_judge


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python -m decisiontracebench {score,judge} [options]')
        return 0 if len(sys.argv) > 1 else 2
    command = sys.argv.pop(1)
    if command == 'score':
        return cli_score.main()
    if command == 'judge':
        return cli_judge.main()
    print(f'Unknown command: {command}; choose score or judge', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

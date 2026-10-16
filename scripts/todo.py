"""Print the internal source-of-truth TODO; never execute experiments."""
from pathlib import Path

if __name__=='__main__':
    print(Path(__file__).with_name('TODO.md').read_text(encoding='utf-8'))

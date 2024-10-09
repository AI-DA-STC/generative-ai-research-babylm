import sys
from normalize import clean
import re


regex_1 = re.compile(r"\[\d+\]")
regex_2 = re.compile(r"\[\[([^\|\]]*)\|*[^\]]*\]\]")
regex_3 = re.compile(r"= = = ([^\=]*) = = =")


def preprocess(f):
    prev_line = None
    for i, line in enumerate(f):
        line = ' '.join(line.strip().split())
        line = clean(line, minimal=True)

        if i > 0 and line.startswith("= = = "):
            yield ""

        if len(line) == 0:
            continue

        if line.startswith("[[Category:") or line.startswith("[[File:"):
            continue
        
        line = regex_1.sub("", line)
        line = regex_2.sub(r"\1", line)
        line = regex_3.sub(r"\1", line)

        yield line

def main(input_path,output_path):
    with open(input_path) as f:
        with open(output_path, 'w') as g:
            for line in preprocess(f):
                g.write(f"{line}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    main(input_path, output_path)
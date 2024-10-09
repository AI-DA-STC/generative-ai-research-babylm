import sys
from normalize import clean
import re
from tqdm import tqdm


regex = re.compile(r"\[\d+\]")


def preprocess(f):
    prev_line = None
    for line in tqdm(f,desc='processing'):
        line = ' '.join(line.strip().split())
        line = regex.sub("", line)
        line = clean(line, minimal=True)

        if len(line) == 0:
            if prev_line is not None and prev_line != "":
                yield prev_line
                yield ""
            prev_line = None
            continue

        if "is a commune. It is" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a commune found" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a city in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a village in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a municipality in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a town in" in line and len(line) < 128:
            prev_line = None
            continue

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        if prev_line is not None:
            yield prev_line

        prev_line = line

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
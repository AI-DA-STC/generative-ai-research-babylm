import sys
from normalize import clean
from tqdm import tqdm


def preprocess(f):
    prev_line = None
    for line in tqdm(f,desc='processing'):
        line = ' '.join(line.strip().split())
        line = clean(line, minimal=True)

        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("-"):
            line = line[1:]

        if len(line) == 0:
            if prev_line is None or len(prev_line) > 0:
                yield ""
            prev_line = line
            continue

        if not line.endswith(":") and not line.startswith('"') and not line.endswith('"') and not (line.startswith("(") and line.endswith(")")) and not (line.startswith("[") and line.endswith("]")) and not (line.startswith("{") and line.endswith("}")):
            line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
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
import sys
from normalize import clean


def preprocess(f):
    prev_line = None
    for line in f:
        line = line.strip()

        line = line[0].upper() + line[1:]
        line = clean(line)
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
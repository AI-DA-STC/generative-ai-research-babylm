import sys
from normalize import clean


def preprocess(f):
    for line in f:
        line = line.strip()

        line = line.replace("-LRB-", "(")
        line = line.replace("-LCB-", "{")
        line = line.replace("-LSB-", "[")
        line = line.replace("-RRB-", ")")
        line = line.replace("-RCB-", "}")
        line = line.replace("-RSB-", "]")

        line = line.replace("`` ", '"')
        line = line.replace("``", '"')
        line = line.replace(" ''", '"')
        line = line.replace("''", '"')

        if len(line) == 0:
            yield ""
            continue

        line = clean(line)

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
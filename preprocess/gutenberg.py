import sys
from normalize import clean
from tqdm import tqdm


def preprocess(f):
    last_num_non_blank_lines = 0
    num_blank_lines = 0
    accumulated_line = []
    for line in tqdm(f,desc='processing'):
        line = ' '.join(line.strip().split())
        line = clean(line, minimal=True)

        if len(line) == 0:
            if len(accumulated_line) > 0:
                yield ' '.join(accumulated_line)
                last_num_non_blank_lines = len(accumulated_line)

            if num_blank_lines == 1 and last_num_non_blank_lines > 1:
                yield ""

            accumulated_line = []
            num_blank_lines += 1
            continue

        num_blank_lines = 0
        accumulated_line.append(line)



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
import sys

def aggregate_output():
    last_line = None
    count = 1

    for line in sys.stdin:
        line = line.strip()  # Remove leading/trailing whitespace
        if line == last_line:
            count += 1
        else:
            if last_line is not None:
                # Print the previous line with its count
                if count > 1:
                    print(f"{last_line} ({count})")
                else:
                    print(last_line)
            last_line = line
            count = 1

    # Print the last line if it exists
    if last_line is not None:
        if count > 1:
            print(f"{last_line} ({count})")
        else:
            print(last_line)

if __name__ == "__main__":
    aggregate_output()

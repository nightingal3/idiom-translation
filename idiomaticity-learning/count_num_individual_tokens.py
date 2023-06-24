import os

if __name__ == "__main__":
    data_path = "./data/"
    #data_path = "/compute/tir-0-15/mengyan3/data"
    corpus_sizes = ["100k", "1M", "10M"]
    num_idioms = ["10", "100", "1k", "10k", "100k", "1M"]
    special_context = True

    for corpus_size in corpus_sizes:
        for num_idiom in num_idioms:
            directory_name = f"corpus_{corpus_size}_idioms_{num_idiom}" + ("_specialcontext" if special_context else "")
            directory_path = os.path.join(data_path, directory_name)
            if not os.path.exists(directory_path):
                continue
            for filename in os.listdir(directory_path):
                if filename == "train.input":
                    print(f"Processing {corpus_size}, {num_idiom}")
                    src_path = os.path.join(directory_path, filename)
                    trg_path = os.path.join(directory_path, filename.replace(".input", ".label"))
                    src_lines = open(src_path).readlines()
                    trg_lines = open(trg_path).readlines()

                    num_0 = 0
                    num_1 = 0
                    num_0_outside_idiom = 0
                    num_1_outside_idiom = 0
                    n_idioms = 0
                    for l in src_lines:
                        line = l.strip().split()
                        num_0 += line.count("0")
                        num_1 += line.count("1")

                        if "0" in line and (line.index("0") == len(line) - 1 or line[line.index("0") + 1] != "1"):
                            num_0_outside_idiom += 1
                        if "1" in line and (line.index("1") == 0 or line[line.index("1") - 1] != "0"):
                            num_1_outside_idiom += 1
                        
                        if "0 1" in l:
                            n_idioms += 1

                    print(f"Num 0: {num_0}")
                    print(f"Num 1: {num_1}")
                    print(f"Num 0 outside idiom: {num_0_outside_idiom}")
                    print(f"Num 1 outside idiom: {num_1_outside_idiom}")    
                    print(f"Total idioms: {n_idioms}")    

                    


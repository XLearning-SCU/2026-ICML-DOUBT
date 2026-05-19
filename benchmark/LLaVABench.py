from datasets import load_dataset


class LLaVABench:

    def __init__(self):
        self.ds = load_dataset("lmms-lab/llava-bench-in-the-wild")

    def obtain_size(self):
        return len(self.ds['train'])

    def retrieve(self, idx):
        row = self.ds['train'][idx]
        result = {
            'idx': idx,
            'img': row['image'],
            'question': row['question'],
            'gt_ans': row['gpt_answer'],
        }
        return result

if __name__ == "__main__":
    benchmark = LLaVABench()
    print(benchmark.retrieve(0))
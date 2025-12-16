import os
import json
import random
import argparse
from typing import List, Tuple, Dict

from openai import OpenAI

# Use OpenRouter API
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# %% load examples
def load_blocks(path: str) -> List[List[str]]:
    """Load blank-line separated blocks from the log file."""
    blocks, block = [], []
    for line in open(path, 'r'):
        if line.strip() == "":
            blocks.append(block)
            block = []
        else:
            if line.strip():
                block.append(line.strip())
    assert len(blocks) % 2 == 0
    return blocks

def remove_invalid_steps(actions: List[str]) -> List[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            try:
                if type(eval(arg)) == str and type(eval(arg[1:-1])) == int:
                    valid_actions.append(a)
            except:
                continue
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
        elif "scroll(" in a or "noop(" in a:
            continue
        else:
            valid_actions.append(a)
    return valid_actions

def extract_think_and_action(path: str) -> Tuple[List[str], List[str]]:
    """Extract the task trajectory from the log file."""
    blocks = load_blocks(path)
    think_list, action_list = [], []
    for i in range(1, len(blocks), 2):
        # action
        b = blocks[i]
        actions = remove_invalid_steps(b[1:])
        if len(actions) == 0: continue
        action_list.append(actions)
        # think
        b = blocks[i-1]
        idx = b[-1].index("browsergym.experiments.loop - INFO -")
        think_list.append(b[-1][idx+36: ].strip())
    
    assert len(think_list) == len(action_list)
    
    # TODO: merge same actions
    return think_list, action_list

def format_trajectory(think_list: List[str], action_list: List[List[str]]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        acts = '\n'.join(a)
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def random_group_sample(d: Dict, n) -> List:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, min(n, len(v)))]

# %% prompt model
def format_examples(examples: List[Dict]) -> str:
    """Format examples to the prompt."""
    formatted_examples = []
    for ex in examples:
        trajectory = format_trajectory(ex["think_list"], ex["action_list"])
        formatted_examples.append(f"Query: {ex['query']}\nActions:\n{trajectory}")
    return '\n\n'.join(["## Concrete Examples"] + formatted_examples + ["## Summary Workflows"])

def llm_generate(examples: List[Dict], args, verbose: bool = False):
    """Call gpt model to generate workflows."""
    prompt = format_examples(examples)
    prompt = '\n\n'.join([args.INSTRUCTION, args.ONE_SHOT, prompt])
    if verbose: print("Prompt:\n", prompt, '\n\n')
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=2048,
    )
    response = response.choices[0].message.content
    if verbose: print(response)
    return response


def main():
    # collect result directories, e.g., ["results/webarena.0", ...]
    args.result_dir = args.result_dir.split()
    if args.criteria == "gt":
        file_dirs = []
        for res_dir in args.result_dir:
            for f in os.listdir(res_dir):
                full_path = os.path.join(res_dir, f)
                summary_path = os.path.join(full_path, "summary_info.json")
                if os.path.isdir(full_path) and os.path.exists(summary_path):
                    try:
                        if json.load(open(summary_path))["cum_reward"]:
                            file_dirs.append(full_path)
                    except:
                        continue
    elif args.criteria == "autoeval":
        file_dirs = []
        for res_dir in args.result_dir:
            for f in os.listdir(res_dir):
                record_path = os.path.join(res_dir, f, f"{args.model}_autoeval.json")
                if not os.path.exists(record_path): continue
                record = json.load(open(record_path))
                if record[0]["rm"]:
                    file_dirs.append(os.path.join(res_dir, f))
    elif args.criteria == "llm_eval":
        # Use LLM evaluation results for string_match tasks
        file_dirs = []
        for res_dir in args.result_dir:
            for f in os.listdir(res_dir):
                # Check task config to determine eval type
                task_id = f.split('_')[0].split(".")[-1] if "_" in f else f.split(".")[-1]
                config_path = os.path.join("config_files", f"{task_id}.json")
                if not os.path.exists(config_path):
                    continue

                config = json.load(open(config_path))
                eval_types = config.get("eval", {}).get("eval_types", [])
                summary_path = os.path.join(res_dir, f, "summary_info.json")

                if not os.path.exists(summary_path):
                    continue

                summary = json.load(open(summary_path))

                # For string_match tasks, use LLM eval if available
                if "string_match" in eval_types:
                    llm_reward = summary.get("llm_reward", 0)
                    if llm_reward > 0:
                        file_dirs.append(os.path.join(res_dir, f))
                else:
                    # For other tasks, use GT reward
                    if summary.get("cum_reward", 0) > 0:
                        file_dirs.append(os.path.join(res_dir, f))
    else:
        raise ValueError(f"Invalid criteria: {args.criteria}.")
    
    print(f"Collected {len(file_dirs)} result directories.")

    # template id based deduplication
    template_dict = {}
    for f in file_dirs:
        # get query -> task objective
        task_id = f.split('/')[-1].split("_")[0].split(".")[1]
        config_path = os.path.join("config_files", f"{task_id}.json")
        config = json.load(open(config_path))
        query = config["intent"]

        template_id = config["intent_template_id"] # for deduplication

        # parse trajectory
        log_path = os.path.join(f, "experiment.log")
        try:
            think_list, action_list = extract_think_and_action(log_path)
        except:
            continue

        # add to template dict
        wdict = {"query": query, "think_list": think_list, "action_list": action_list}
        if template_id not in template_dict: template_dict[template_id] = []
        template_dict[template_id].append(wdict)
    selected_examples = random_group_sample(template_dict, args.num_samples)
    print(f"#{len(selected_examples)} result dirs after template dedup..")
    
    # prompt model to induce workflows
    workflows = llm_generate(selected_examples, args, True)
    workflows += "\n\nclick('id') # input string id value for all actions\n\nselect_option('id', 'value') # for dropdown menu"

    if args.output_path is None:
        website = config["sites"][0]  # assumes all results are about the same website
        args.output_path = f"workflow/{website}_neural.txt"
        print(f"[Warning] no output path specified, using '{args.output_path}' by default")
        
    with open(args.output_path, 'w') as fw:
        fw.write(workflows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Path to the result directory. Support multiple directories separated by space.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to the output file.")
    parser.add_argument("--criteria", type=str, default="llm_eval",
                        choices=["gt", "autoeval", "llm_eval"],
                        help="'gt': only use examples with gold reward, 'autoeval': use examples with autoeval reward, 'llm_eval': use LLM eval for string_match tasks, GT for others.")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash-preview-09-2025",
                        help="Model to use for workflow generation (via OpenRouter)")
    parser.add_argument("--num_samples", type=int, default=1, help="Max number of samples to input per template.")
    args = parser.parse_args()

    args.INSTRUCTION = open("prompt/instruction.txt", 'r').read()
    args.ONE_SHOT = open("prompt/one_shot.txt", 'r').read()

    main()

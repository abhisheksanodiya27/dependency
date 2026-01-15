# import os

# def project_tree(root, level=0, max_depth=4):
#     if level > max_depth:
#         return

#     for item in os.listdir(root):
#         path = os.path.join(root, item)
#         print("│   " * level + "├── " + item)

#         if os.path.isdir(path):
#             project_tree(path, level + 1, max_depth)

# project_tree(r"C:\Users\ABHISHEK SANODIYA\Desktop\Dependency\Dependency_LLM\NorthwindTraders")
import os

EXCLUDE_DIRS = {"venv", "__pycache__", ".git", "node_modules"}

def generate_project_tree(root=".", max_depth=5, level=0):
    if level > max_depth:
        return ""

    tree = ""
    for item in sorted(os.listdir(root)):
        if item in EXCLUDE_DIRS:
            continue

        path = os.path.join(root, item)
        tree += "│   " * level + "├── " + item + "\n"

        if os.path.isdir(path):
            tree += generate_project_tree(path, max_depth, level + 1)

    return tree


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(project_tree, file_path, file_code):
    return f"""
You are a senior software architect.

Your task:
1. Analyze the given file
2. Identify all imported internal dependencies
3. Match them against the provided project tree
4. Return the most probable file paths
5. Clearly separate internal vs external dependencies

Rules:
- Use ONLY the provided project tree
- Do NOT hallucinate files
- If a dependency is ambiguous, say so
- Output JSON ONLY

PROJECT TREE:
----------------
{project_tree}

TARGET FILE PATH:
-----------------
{file_path}

TARGET FILE CODE:
-----------------
{file_code}

OUTPUT FORMAT (JSON):
{{
  "internal_dependencies": [
    {{
      "import": "services.user_service",
      "probable_path": "services/user_service.py",
      "confidence": "high"
    }}
  ],
  "external_dependencies": [
    "fastapi",
    "sqlalchemy"
  ],
  "unresolved_imports": []
}}
"""

from openai import OpenAI
import json

client = OpenAI(api_key="")

def analyze_dependencies(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise code analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def main():
    project_root = r"C:\Users\ABHISHEK SANODIYA\Desktop\Dependency\Dependency_LLM\NorthwindTraders"
    target_file = r"C:\Users\ABHISHEK SANODIYA\Desktop\Dependency\Dependency_LLM\NorthwindTraders\Src\Domain\Entities\Employee.cs"  # change this

    tree = generate_project_tree(project_root)
    code = read_file(target_file)
    prompt = build_prompt(tree, target_file, code)

    result = analyze_dependencies(prompt)
    print("\n===== LLM OUTPUT =====\n")
    print(result)

if __name__ == "__main__":
    main()
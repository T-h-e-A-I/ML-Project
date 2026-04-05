import os
from collections import defaultdict

# Configuration
root_dir = "./src/ingestion"
output_file = "git_ingest.txt"
ignore_dirs = {}

# --- Filter Settings ---
tree_allowed_extensions = {}          # Show ONLY these in tree (set None to allow all)
tree_ignore_extensions  = {".jpeg", ".png"}     # Hide these from tree
content_allowed_extensions = {".py"}   # Include ONLY these in content (set None to allow all)
content_ignore_extensions  = {".jpeg", ".json", ".png"}        # Exclude these from content

def is_visible(filename, allowed, ignored):
    ext = os.path.splitext(filename)[1]
    if ignored and ext in ignored:
        return False
    if allowed and ext not in allowed:
        return False
    return True

def generate_tree(dir_path, prefix=""):
    tree = ""
    entries = sorted(os.listdir(dir_path))

    for entry in entries:
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):
            if entry in ignore_dirs:
                continue
            subtree, ignored_summary = generate_tree_with_summary(full_path, prefix + "│   ")
            if subtree or ignored_summary:
                tree += f"{prefix}├── {entry}/\n"
                tree += subtree
                if ignored_summary:
                    for ext, count in sorted(ignored_summary.items()):
                        tree += f"{prefix}│ {count} file{'s' if count > 1 else ''} ({ext})\n"
        else:
            if not is_visible(entry, tree_allowed_extensions, tree_ignore_extensions):
                continue
            tree += f"{prefix}├── {entry}\n"

    return tree

def generate_tree_with_summary(dir_path, prefix=""):
    """Returns (tree_str, ignored_counts_dict) for a directory."""
    tree = ""
    ignored_counts = defaultdict(int)
    entries = sorted(os.listdir(dir_path))

    for entry in entries:
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):
            if entry in ignore_dirs:
                continue
            subtree, sub_ignored = generate_tree_with_summary(full_path, prefix + "│   ")
            if subtree or sub_ignored:
                tree += f"{prefix}├── {entry}/\n"
                tree += subtree
                if sub_ignored:
                    for ext, count in sorted(sub_ignored.items()):
                        tree += f"{prefix}│ {count} file{'s' if count > 1 else ''} ({ext})\n"
        else:
            ext = os.path.splitext(entry)[1]
            if not is_visible(entry, tree_allowed_extensions, tree_ignore_extensions):
                ignored_counts[ext if ext else "(no extension)"] += 1
            else:
                tree += f"{prefix}├── {entry}\n"

    return tree, ignored_counts

def generate_code_doc(out, root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

        for filename in filenames:
            if not is_visible(filename, content_allowed_extensions, content_ignore_extensions):
                continue

            filepath = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(filepath, root_dir)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                out.write(f"\n\n## file name: `{rel_path}`\n\n")
                out.write("```" + os.path.splitext(filename)[1].lstrip(".") + "\n")
                out.write(content)
                out.write("\n```\n")

            except Exception as e:
                print(f"⚠️ Skipping {rel_path}: {e}")

# Run script
with open(output_file, "w", encoding="utf-8") as out:
    out.write("# 📂 Project Directory Tree\n\n")

    # Build root-level tree with summary
    root_tree, root_ignored = generate_tree_with_summary(root_dir)
    full_tree = root_tree
    if root_ignored:
        for ext, count in sorted(root_ignored.items()):
            full_tree += f" {count} file{'s' if count > 1 else ''} ({ext})\n"

    out.write("```\n" + full_tree + "```\n")
    out.write("\n\n# 📄 Code Contents\n")
    generate_code_doc(out, root_dir)

print(f"\n✅ Ingest file generated: {output_file}")
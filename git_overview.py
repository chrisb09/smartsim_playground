#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse

# ANSI Colors
C_BRANCH = "\033[1;34m"
C_AHEAD = "\033[1;32m"
C_BEHIND = "\033[1;31m"
C_STAGED = "\033[1;32m"
C_MODIFIED = "\033[1;33m"
C_UNTRACKED = "\033[1;31m"
C_SYNC = "\033[1;35m"
C_URL = "\033[0;36m"
C_RESET = "\033[0m"
C_DIM = "\033[2m"

def run_git(args, cwd=None):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return None

def get_line_stats(path, filename, staged=False):
    args = ["diff"]
    if staged:
        args.append("--cached")
    args.extend(["--numstat", "--", filename])
    
    out = run_git(args, path)
    if out:
        parts = out.split()
        if len(parts) >= 2:
            add, delt = parts[0], parts[1]
            if add == "-" or delt == "-":
                return "binary"
            if add == "0" and delt == "0":
                return ""
            return f"+{add} -{delt}"
    return ""

def get_git_info(path, verbose=False):
    info = {}
    
    # Branch
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], path)
    if branch is None: return None
    info['branch'] = branch

    # Remote URL
    remote_url = run_git(["remote", "get-url", "origin"], path)
    if not remote_url:
        remotes = run_git(["remote"], path)
        if remotes:
            first_remote = remotes.splitlines()[0]
            remote_url = run_git(["remote", "get-url", first_remote], path)
    info['url'] = remote_url

    # Remote Tracking
    upstream = run_git(["rev-parse", "--abbrev-ref", "@{u}"], path)
    if upstream:
        diff = run_git(["rev-list", "--left-right", "--count", f"HEAD...{upstream}"], path)
        if diff:
            ahead, behind = diff.split()
            info['ahead'] = int(ahead)
            info['behind'] = int(behind)
            info['upstream'] = upstream
    else:
        info['ahead'] = 0
        info['behind'] = 0

    # Status
    status_out = run_git(["status", "--porcelain"], path)
    info['staged_files'] = []
    info['modified_files'] = []
    info['untracked_files'] = []
    
    if status_out:
        for line in status_out.splitlines():
            if len(line) < 4: continue
            index_status = line[0]
            worktree_status = line[1]
            
            # The path starts at index 3 in porcelain format
            fname = line[3:].strip('"')
            
            if index_status in "MADRC":
                stats = get_line_stats(path, fname, staged=True) if verbose else ""
                info['staged_files'].append((fname, stats))
            if worktree_status in "MAD":
                stats = get_line_stats(path, fname, staged=False) if verbose else ""
                info['modified_files'].append((fname, stats))
            if index_status == "?" or worktree_status == "?":
                info['untracked_files'].append(fname)
    
    info['staged_count'] = len(info['staged_files'])
    info['modified_count'] = len(info['modified_files'])
    info['untracked_count'] = len(info['untracked_files'])

    return info

def get_submodules_recursive(path):
    out = run_git(["submodule", "status", "--recursive"], path)
    submodules = []
    if out:
        for line in out.splitlines():
            if not line: continue
            status_char = line[0]
            parts = line[1:].strip().split()
            commit_hash = parts[0]
            sub_path = parts[1]
            
            submodules.append({
                'path': sub_path,
                'is_clean_in_super': status_char == ' ',
                'hash': commit_hash
            })
    return submodules

def format_status(info):
    status_str = f"{C_BRANCH}{info['branch']}{C_RESET}"
    
    if info['ahead'] > 0 or info['behind'] > 0:
        status_str += f" [{C_AHEAD}↑{info['ahead']}{C_RESET} {C_BEHIND}↓{info['behind']}{C_RESET}]"
    
    changes = []
    if info['staged_count'] > 0: changes.append(f"{C_STAGED}{info['staged_count']} staged{C_RESET}")
    if info['modified_count'] > 0: changes.append(f"{C_MODIFIED}{info['modified_count']} modified{C_RESET}")
    if info['untracked_count'] > 0: changes.append(f"{C_UNTRACKED}{info['untracked_count']} untracked{C_RESET}")
    
    if changes:
        status_str += " (" + ", ".join(changes) + ")"
    return status_str

def main():
    parser = argparse.ArgumentParser(description="Tree-like overview of a nested git repository and its submodules.")
    parser.add_argument("path", nargs="?", default=".", help="Root repository path (default: current directory)")
    parser.add_argument("-v", "--verbose", action="store_true", help="List changed files with line statistics and remote URLs")
    args = parser.parse_args()

    root_abs = os.path.abspath(args.path)
    if not os.path.isdir(os.path.join(root_abs, ".git")):
        print(f"Error: {root_abs} is not a git repository.")
        sys.exit(1)

    all_subs = get_submodules_recursive(root_abs)
    sub_map = {sub['path']: sub for sub in all_subs}
    sorted_paths = sorted(sub_map.keys())

    def get_direct_children(parent_path):
        children = []
        for p in sorted_paths:
            if parent_path:
                if p.startswith(parent_path + "/"):
                    relative = p[len(parent_path)+1:]
                    is_direct = True
                    for other in sorted_paths:
                        if other != p and other.startswith(parent_path + "/") and p.startswith(other + "/"):
                            is_direct = False
                            break
                    if is_direct: children.append(p)
            else:
                is_direct = True
                for other in sorted_paths:
                    if other != p and p.startswith(other + "/"):
                        is_direct = False
                        break
                if is_direct: children.append(p)
        return children

    def print_node(path, name, prefix, is_last, is_root=False):
        connector = "" if is_root else ("└── " if is_last else "├── ")
        
        info = get_git_info(path, verbose=args.verbose)
        if not info:
            print(f"{prefix}{connector}{name} [Git Error]")
            return

        status_line = format_status(info)
        sync_str = ""
        if not is_root:
            sub_info = sub_map.get(os.path.relpath(path, root_abs))
            if sub_info and not sub_info['is_clean_in_super']:
                sync_str = f" {C_SYNC}[OUT OF SYNC WITH SUPER]{C_RESET}"

        print(f"{prefix}{connector}{name}: {status_line}{sync_str}")
        
        rel_path = "" if is_root else os.path.relpath(path, root_abs)
        children = get_direct_children(rel_path)
        
        if args.verbose:
            inner_prefix = prefix if is_root else (prefix + ("    " if is_last else "│   "))
            # We use a vertical line for attributes if there are more things (children) coming
            attr_prefix = inner_prefix + ("│   " if children else "    ")
            
            if info['url']:
                print(f"{attr_prefix}{C_DIM}URL: {C_URL}{info['url']}{C_RESET}")
            
            for fname, stats in info['staged_files']:
                stat_str = f" {C_DIM}({stats}){C_RESET}" if stats else ""
                print(f"{attr_prefix}  {C_STAGED}[S]{C_RESET} {fname}{stat_str}")
            for fname, stats in info['modified_files']:
                stat_str = f" {C_DIM}({stats}){C_RESET}" if stats else ""
                print(f"{attr_prefix}  {C_MODIFIED}[M]{C_RESET} {fname}{stat_str}")
            for fname in info['untracked_files']:
                print(f"{attr_prefix}  {C_UNTRACKED}[U]{C_RESET} {fname}")

        # Recurse
        new_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")
        for i, child_path in enumerate(children):
            parent_rel = rel_path + "/" if rel_path else ""
            display_name = child_path[len(parent_rel):]
            print_node(os.path.join(root_abs, child_path), display_name, new_prefix, i == len(children)-1)

    root_name = os.path.basename(root_abs)
    print_node(root_abs, root_name, "", True, is_root=True)

if __name__ == "__main__":
    main()

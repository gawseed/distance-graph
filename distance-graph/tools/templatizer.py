import os
import re
import json
import collections

TOKENIZE_PATTERN = re.compile('[\s"=;|&><]')
_NIX_COMMANDS = None

## command templatizer code 
def is_bash_directive(s, stopwords):
  """Returns True if s is bash syntax (e.g., fi), standard command (e.g., wget), or looks like an argument (e.g., -c).
  """
  if s.startswith('-'): return True
  global _NIX_COMMANDS
  if _NIX_COMMANDS is None:
    _NIX_COMMANDS = set(open(stopwords).read().split('\n'))
    # _NIX_COMMANDS = set(open('../example/nix_commands.txt').read().split('\n'))
  return s in _NIX_COMMANDS

class CommandNode:
  """Command node is a graph on its own.

  cmd_arr is an array of commands run
  """
  def __init__(self, cmd_arr, stopwords):
    self.basename2tok_ids = collections.defaultdict(list)
    self.cmd_str = str(cmd_arr)
    all_tokens = []
    for cmd_entry in cmd_arr:
      sub_tokens = TOKENIZE_PATTERN.split(cmd_entry)
      sub_tokens = [t for t in sub_tokens if t]
      all_tokens += sub_tokens
    all_tokens = tuple(all_tokens)
    self.all_tokens = all_tokens

    # Make sets out of path parts.
    for i, tok in enumerate(all_tokens):
      if is_bash_directive(tok, stopwords):
        continue
      basename = os.path.basename(tok)
      self.basename2tok_ids[basename].append(i)


class CommandNodeEval(CommandNode): ## declare that CommandNodeEval class inherits from CommandNode
  """Command array string gets tokenized."""
  def __init__(self, encoded_command_str, stopwords):
    super().__init__(eval(encoded_command_str), stopwords)


class CommandNodeJson(CommandNode):
  """Input object is a json encoded array to convert to a command array""" 
  def __init__(self, json_command_str):
    super().__init__(json.loads(json_command_str))


def _basename_template(basename, command_node):
  token_ids = set(command_node.basename2tok_ids[basename])
  template_tokens = []
  for tid, token in enumerate(command_node.all_tokens):
    if tid in token_ids:
      template_tokens.append(os.path.join(os.path.dirname(token), '%0'))
    else:
      template_tokens.append(token)

  template = ('basename', '%0', tuple(template_tokens))
  return template

def possible_templates(command_node):
  # Basename
  for basename in command_node.basename2tok_ids.keys():
    # templetize basename
    yield _basename_template(basename, command_node)


class CommandGraph:

  def __init__(self):
    self.command_nodes = []
    self.template2commands = collections.defaultdict(list)  # Edges template -> command nodes
  
  def add(self, cn):
    # Step 1: create command node. It creates sets of basenames.
    self.command_nodes.append(cn)
    
    for template in possible_templates(cn):
      self.template2commands[template].append(cn)

  def finalize_degrees(self):
    self.template2degree = {t: len(cmds) for t, cmds in self.template2commands.items()}

  def write_debug_sharded_json(self, output_prefix, entries_per_shard=1000, num_examples_per_template=100):
    import json
    import random
    """Writes template to sample of commands."""
    dirname = os.path.dirname(output_prefix)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self.finalize_degrees()
    sorted_by_degree = sorted(self.template2degree.items(), key=lambda k: -k[1])

    next_file_id = 0
    entries = []
    for i, (template, degree) in enumerate(sorted_by_degree):
      if degree <= 1:
        continue
      # Shallow copy
      commands = [x for x in self.template2commands[template]]
      random.shuffle(commands)
      entries.append({
        'template': template,
        'num_commands': len(commands),
        'commands': [c.cmd_str for c in commands[:entries_per_shard]],
      })
      if len(entries) >= entries_per_shard:
        # Write
        with open(output_prefix + ('_%i.json' % next_file_id), 'w') as fout:
          fout.write(json.dumps(entries))
        entries = []
        next_file_id += 1

    if len(entries) >= 1:
      # Write
      with open(output_prefix + ('_%i.json' % next_file_id), 'w') as fout:
        fout.write(json.dumps(entries))
    print('wrote %i jsons with prefix %s' % (next_file_id, output_prefix))

  def cmd_to_template(self):
    self.finalize_degrees()
    processed_cmds = set()
    sorted_by_degree = sorted(self.template2degree.items(), key=lambda k: -k[1])
    command2template = {}
    for i, (template, degree) in enumerate(sorted_by_degree):
      if degree <= 1: continue
      for cmd in self.template2commands[template]:
        if cmd in processed_cmds:
          continue
        command2template[cmd.cmd_str] = template 
        processed_cmds.add(cmd)
    return command2template
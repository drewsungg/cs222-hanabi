# generative agent utils
# cqz@cs.stanford.edu

# last updated: october 2024

import os
from dotenv import load_dotenv
import numpy as np
import pickle
import pandas as pd
import json
import re

from openai import OpenAI
from anthropic import Anthropic 

load_dotenv()

oai = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

ant = Anthropic()
ant.api_key = os.getenv('ANTHROPIC_API_KEY')

def gen_oai(messages, model='gpt-4o', temperature=1):
  if model == None:
    model = 'gpt-4o'
  try:
    response = oai.chat.completions.create(
      model=model,
      temperature=temperature,
      messages=messages,
      max_tokens=500)
    content = response.choices[0].message.content
    return content
  except Exception as e:
    print(f"Error generating completion: {e}")
    raise e

def simple_gen_oai(prompt, model='gpt-4o', temperature=1):
  messages = [{"role": "user", "content": prompt}]
  return gen_oai(messages, model)

def gen_ant(messages, model='claude-3-5-sonnet-20240620', temperature=1, 
            max_tokens=1000):
  if model == None:
    model = 'claude-3-5-sonnet-20240620'
  try:
    response = ant.messages.create(
      model=model,
      max_tokens=max_tokens,
      temperature=temperature,
      messages=messages
    )
    content = response.content[0].text
    return content
  except Exception as e:
    print(f"Error generating completion: {e}")
    raise e

def simple_gen_ant(prompt, model='claude-3-5-sonnet-20240620'):
  messages = [{"role": "user", "content": prompt}]
  return gen_ant(messages, model)

# Prompt utils

# Prompt inputs
def fill_prompt(prompt, placeholders):
  for placeholder, value in placeholders.items():
    placeholder_tag = f"!<{placeholder.upper()}>!"
    if placeholder_tag in prompt:
      prompt = prompt.replace(placeholder_tag, str(value))
  return prompt

def make_output_format(modules):
  '''
  given some module names in the form (name)
  get the module from a file?
  - would be like name, prompt text

  make the description of the output format (json for now)

  assumes all module outputs are strings
  '''
  output_format = "Provide your response in the following JSON format:\n\n{"
  for module in modules:
    output_format += f"\n\t\"{module['name']}\": \"[{module['description']}]\""
    if module != modules[-1]:
      output_format += ","
  output_format += "\n}"
  return output_format

def modular_instructions(modules):
  '''
  given some module names in the form (name)
  get the module from a file?
  - would be like name, prompt text

  make the whole prompt
  '''
  prompt = ""
  for i, module in enumerate(modules):
    prompt += f"Step {i + 1} ({module['name']}): {module['instruction']}\n"
  prompt += "\n"
  prompt += make_output_format(modules)
  return prompt

# Prompt outputs
def parse_json(response, target_keys=None):
  json_start = response.find('{')
  json_end = response.rfind('}') + 1
  cleaned_response = response[json_start:json_end].replace('\\"', '"')
  
  try:
    parsed = json.loads(cleaned_response)
    if target_keys:
      parsed = {key: parsed.get(key, "") for key in target_keys}
    return parsed
  except json.JSONDecodeError:
    print("Tried to parse json, but it failed. Switching to regex fallback.")
    parsed = {}
    for key_match in re.finditer(r'"(\w+)":\s*', cleaned_response):
      key = key_match.group(1)
      if target_keys and key not in target_keys:
        continue
      value_start = key_match.end()
      if cleaned_response[value_start] == '"':
        value_match = re.search(r'"(.*?)"(?:,|\s*})', 
                                cleaned_response[value_start:])
        if value_match:
          parsed[key] = value_match.group(1)
      elif cleaned_response[value_start] == '{':
        nested_json = re.search(r'(\{.*?\})(?:,|\s*})', 
                                cleaned_response[value_start:], re.DOTALL)
        if nested_json:
          try:
            parsed[key] = json.loads(nested_json.group(1))
          except json.JSONDecodeError:
            parsed[key] = {}
      else:
        value_match = re.search(r'([^,}]+)(?:,|\s*})', 
                                cleaned_response[value_start:])
        if value_match:
          parsed[key] = value_match.group(1).strip()
    
    if target_keys:
      parsed = {key: parsed.get(key, "") for key in target_keys}
    return parsed
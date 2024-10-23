import os
import random
import io
from flask import Flask, render_template, jsonify, request, send_file
from agents import agent_list
from llm_utils import *

class Agent:
  def __init__(self, name, persona):
    self.name = name
    self.persona = persona
    self.messages = []

class Game:
  def __init__(self, agents):
    self.agents = agents
    self.position = "group leader"
    self.public_messages = []
    self.round_number = 0
    self.winner = ""
    self.gamestate = "Nothing has been said yet. Start the conversation. You don't know anything about the other people in the group yet, and vice versa.\n"
    self.log = ""

  def update_gamestate(self, agent_name, message):
    self.public_messages.append(f"{agent_name}: {message}")
    self.gamestate = "START OF CONVERSATION SO FAR.\n" + "\n".join(self.public_messages) + "\nEND OF CONVERSATION SO FAR."

  def instruct_agent(self, agent, instruction):
    system_prompt = self._create_system_prompt(agent)
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": self.gamestate},
      {"role": "user", "content": instruction}
    ]
    return gen_oai(messages)

  def _create_system_prompt(self, agent):
    return f"""
    YOU: You are {agent.name}, {agent.persona}. Speak in character as {agent.name} with very short messages in a conversational tone. 
    
    SCENARIO: You are in a group of people containing {', '.join(a.name for a in self.agents)}. You are meeting everyone else for the first time. Your group is trying to decide who should be the {self.position}. At the end of the discussion, each person will vote for one person in the group other than themselves.
    
    STYLE: Write in the style of someone texting, with short messages and minimal punctuation. No emojis. Speak in your own personal voice. Don't use generic or vague language."""

  def get_agent_response(self, agent, current_round, total_rounds):
    modules = self._get_modules_for_round(current_round, total_rounds)
    target_keys = [module["name"] for module in modules]

    instruction = modular_instructions(modules)
    response = self.instruct_agent(agent, instruction)
    parsed = parse_json(response, target_keys=target_keys)

    agent_data = {"name": agent.name}
    for key in target_keys:
      if key in parsed:
        agent_data[key] = parsed[key]
        print(f"{agent.name} {key.upper()}: {parsed[key]}")
        print()

    if "message" in parsed:
      self.update_gamestate(agent.name, parsed["message"])

    self._update_log(agent_data, current_round)
    return agent_data

  def _get_modules_for_round(self, current_round, total_rounds):
    if current_round == 1:
      return [self.intro, self.message]
    elif current_round == total_rounds:
      return [self.vote_plan, self.vote]
    else:
      return [self.reflect, self.plan, self.message]

  def _update_log(self, agent_data, current_round):
    if current_round != self.round_number:
      self.round_number = current_round
      self.log += f"\n\n## Round {current_round}\n\n"

    self.log += f"### {agent_data['name']}\n\n"
    for key, value in agent_data.items():
      if key != "name":
        self.log += f"**{key.capitalize()}**: {value}\n\n"

  def run_round(self, current_round, total_rounds):
    round_data = []
    modules = self._get_modules_for_round(current_round, total_rounds)
    target_keys = [module["name"] for module in modules]

    shuffled_agents = self.agents[:]
    random.shuffle(shuffled_agents)
    for agent in shuffled_agents:
      print("=" * 20)
      placeholders = {"AGENT_NAME": agent.name,
                      "POSITION": self.position}
      instruction = modular_instructions(modules)
      prompt = fill_prompt(instruction, placeholders)
      response = self.instruct_agent(agent, prompt)
      parsed = parse_json(response, target_keys=target_keys)

      agent_data = {"name": agent.name}
      for key in target_keys:
        if key in parsed:
          agent_data[key] = parsed[key]
          print(f"{agent.name} {key.upper()}: {parsed[key]}")
          print()

      if "message" in parsed:
        self.update_gamestate(agent.name, parsed["message"])
      
      round_data.append(agent_data)

    if current_round == total_rounds:
      return self._process_voting_results(round_data)

    print(f"Moving to next round. Current round: {current_round}")
    return round_data, None, None

  def _process_voting_results(self, round_data):
    vote_results = {agent.name: 0 for agent in self.agents}
    vote_list = []
    for agent_data in round_data:
      vote = agent_data.get("vote")
      vote_results[vote] += 1
      vote_list.append((agent_data["name"], vote))

    winner = max(vote_results, key=vote_results.get)
    self.winner = winner
    
    print("\nVoting Results:")
    print("-" * 20)
    for name, votes in vote_results.items():
      print(f"{name}: {votes} votes")
    print("-" * 20)
    print(f"The winner is {winner} with {vote_results[winner]} votes!")
    
    return round_data, winner, vote_list

  def log_user_agent(self, name, persona):
    self.log = f"# Game Log\n\n## User-defined Agent\n\n```python\n{{'name': '{name}', 'persona': '{persona}'}}\n```\n"

  def log_voting_round(self, round_data, vote_results, winner):
    self.log += f"\n\n## Round {self.round_number} (Voting)\n\n"
    for agent_data in round_data:
      self.log += f"### {agent_data['name']}\n\n"
      self.log += f"**Plan**: {agent_data['vote_plan']}\n\n"
      self.log += f"**Vote**: {agent_data['vote']}\n\n"

    self.log += "\n## Voting Results\n\n"
    for name, votes in vote_results.items():
      self.log += f"- {name}: {votes} votes\n"
    self.log += f"\n**Winner**: {winner}\n"

  def get_log(self):
    return self.log

  intro = {
    "name": "introduction",
    "instruction": "Because the conversation has just started, everyone needs to introduce themselves. Create a plan for a compelling introduction. Be creative in order to craft the most strategic introduction.",
    "description": "your introduction plan",
  }

  reflect = {
    "name": "reflection",
    "instruction": "Reflect on the situation by answering each of the following questions.\n1] Knowledge: What do you know so far about the others?\n2] Support: You can advocate either for yourself or another person to be the !<POSITION>! -- compare the pros and cons of everyone in the group (including yourself) and decide who in the group seems like the best candidate based on what you know so far. If not much has been said yet, you can ask others to share their answers to a question that would help you make your decision. Make your decision independently of who others might be advocating for. In this step include a sentence saying 'I think the best candidate is...' or 'I'm not sure yet, so I'll ask about...'",
    "description": "your reflection",
  }

  plan = {
    "name": "plan",
    "instruction": "Based on your reflection, write a plan for how you will either gather more information or persuade the others to vote for your preferred candidate. Don't try to respond to everyone at once: your message should be a focused argument that will help you gather more information or persuade the group to vote for your preferred candidate. Remember that only one person can win.",
    "description": "your plan",
  }

  message = {
    "name": "message",
    "instruction": "Write your 2-4 sentence message to the group, incorporating your plan from above. Make sure your message is RESPONSIVE: reference and respond to what has previously been said, answer any questions that were directed towards you, and make sure the conversation flows naturally.",
    "description": "your message",
  }

  vote_plan = {
    "name": "vote_plan",
    "instruction": "The conversation has ended, and it is time to think about who you will vote for. Think step-by-step: first list the pros and cons of each person (excluding yourself, as you cannot vote for yourself), then compare them against the things you value most. Provide your reasoning in this step.",
    "description": "your vote",
  }

  vote = {
    "name": "vote",
    "instruction": "The conversation has ended. Write a vote for the person you think should be the !<POSITION>!. You cannot vote for yourself, even if others support you. Respond with ONLY the name of the person you vote for.",
    "description": "your vote",
  }

def init_game(agents=[]):
  initialized_agents = [Agent(agent_data["name"], agent_data["persona"]) for agent_data in agents]
  return Game(initialized_agents)

app = Flask(__name__)
game = None
current_agent_index = 0
game_data = []

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/add_agent', methods=['POST'])
def add_agent():
  global game
  data = request.json
  new_agent = {"name": data['name'], "persona": data['persona']}
  agent_list.append(new_agent)
  game = init_game(agent_list)
  game.position = data['position']
  game.log_user_agent(data['name'], data['persona'])
  return jsonify({"status": "success"})

@app.route('/next_agent', methods=['POST'])
def next_agent():
  global current_agent_index, game_data
  data = request.json
  current_round = data['current_round']
  total_rounds = data['total_rounds']

  if current_round < total_rounds:
    if current_agent_index == 0:
      # Shuffle the agents at the beginning of each round
      last_speaker = game.agents[-1] if game_data else None
      remaining_agents = [agent for agent in game.agents if agent != last_speaker]
      random.shuffle(remaining_agents)
      
      if last_speaker:
        insert_position = random.randint(1, len(remaining_agents))
        remaining_agents.insert(insert_position, last_speaker)
        game.agents = remaining_agents
      else:
        game.agents = remaining_agents

      if current_round > 1:
        game_data = []

    agent = game.agents[current_agent_index]
    agent_data = game.get_agent_response(agent, current_round, total_rounds)
    game_data.append(agent_data)
    current_agent_index += 1

    round_finished = current_agent_index == len(game.agents)
    if round_finished:
      current_agent_index = 0
      current_round += 1

    return jsonify({
      "agent_data": agent_data, 
      "current_round": current_round,
      "round_finished": round_finished
    })
  elif current_round == total_rounds:
    round_data, winner, vote_list = game.run_round(current_round, total_rounds)
    vote_results = {agent.name: sum(1 for vote in vote_list if vote[1] == agent.name) for agent in game.agents}
    game.log_voting_round(round_data, vote_results, winner)
    return jsonify({
      "finished": True, 
      "winner": winner, 
      "votes": vote_list, 
      "round_data": round_data
    })
  else:
    return jsonify({"finished": True})

original_agent_count = len(agent_list)

@app.route('/reset', methods=['POST'])
def reset_game():
  global game, current_agent_index, game_data, agent_list
  agent_list = agent_list[:original_agent_count]
  game = None
  current_agent_index = 0
  game_data = []
  return jsonify({"status": "reset"})

@app.route('/download_log', methods=['GET'])
def download_log():
  if game:
    log_content = game.get_log()
    buffer = io.BytesIO()
    buffer.write(log_content.encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='game_log.md', mimetype='text/markdown')
  else:
    return jsonify({"error": "No game log available"}), 400

if __name__ == "__main__":
  app.run(debug=True)

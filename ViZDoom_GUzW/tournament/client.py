''' Client (player) connects to the host using the -join parameter.
When all players are connected, client will start playing the game.
'''

from argparse import ArgumentParser
from time import time

import numpy as np
import vizdoom as vzd

from agent_example import Agent

parser = ArgumentParser("ViZDoom training on deathmatch map.")
parser.add_argument("-c", "--colorset", type=int, default=0)
parser.add_argument("-n", "--playername", default="Janusz", help="Name of a player.")
args = parser.parse_args()
colorset = args.colorset
playername = args.playername

game = vzd.DoomGame()
game.load_config("scenarios/deathmatch.cfg")

game.add_game_args(
    "-join 127.0.0.1 -port 5029 "
    "+viz_connect_timeout 5 "
)

game.add_game_args(f"+name {playername} +colorset {colorset}")
game.set_mode(vzd.Mode.PLAYER)
game.init()

actions = np.eye(game.get_available_buttons_size()).tolist()

agent = Agent(actions)

# Get player's number
player_number = int(game.get_game_variable(vzd.GameVariable.PLAYER_NUMBER))
last_frags = 0
players_nb = game.get_server_state().player_count

iterations = 0
duration_print_iters = 100

while not game.is_episode_finished():
    if players_nb != game.get_server_state().player_count:
        break

    game_state = game.get_state()
    
    # Record duration of choosing an action
    t1 = time()

    # Choosing an action
    action, action_index = agent.choose_action(game_state)
    
    if iterations % duration_print_iters == 0:
        print(f"Choosing action duration: {time() - t1:.1f} s")
    
    game.make_action(action)

    frags = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
    if frags != last_frags:
        last_frags = frags
        print(f"Player {player_number} has {frags} frags.")

    if game.is_episode_finished():
        break

    if game.is_player_dead():
        print("Player died.")
        game.respawn_player()
    iterations += 1

game.close()

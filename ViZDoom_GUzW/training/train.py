from argparse import ArgumentParser

import numpy as np
import vizdoom as vzd

from agent_example import Agent, update_config
from utils import display_buffers

train = False

parser = ArgumentParser("ViZDoom training on deathmatch map.")
parser.add_argument("-m", "--map", default="map01", help="Use map01 or map03.")
args = parser.parse_args()
map_name = args.map

if map_name not in ["map01", "map03"]:
    raise ValueError("Please use map01 or map03.")

game = vzd.DoomGame()
game.load_config("scenarios/deathmatch.cfg")
game.set_doom_map(map_name)

game.add_game_args(
    "-host 1 "
    "" if map_name[-1] == "1" else "-deathmatch "  # Deathmatch mode does not apply on map01
    "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
    "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
    "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
    "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
    "+sv_nocrouch 1 "  # Disables crouching.
    "+viz_respawn_delay 2 "  # Sets delay between respawns (in seconds).
)

game.add_game_args("+viz_bots_path scenarios/bots.cfg")
game.add_game_args("+name AI +colorset 0")
game.set_mode(vzd.Mode.PLAYER)
game.set_ticrate(35)
game.set_console_enabled(True)

update_config(game)
if train:
    game.set_window_visible(False)

game.init()

actions = np.eye(game.get_available_buttons_size()).tolist()
last_frags = 0
bots = 5
episodes = 25_000

### DEFINE YOUR AGENT HERE (or init)
agent = Agent(actions)
agent.load_weights("model.pth")

iteration = 0
for i in range(episodes):
    print("Episode #" + str(i + 1))

    ### Add specific number of bots
    # edit this file to adjust bots).
    game.send_game_command("removebots")
    for i in range(bots):
        game.send_game_command("addbot")

    ### Change the bots difficulty
    # Valid args: 1, 2, 3, 4, 5 (1 - easy, 5 - very hard)
    game.send_game_command("pukename change_difficulty 5")

    ### Change number of monster to spawn 
    # Valid args: >= 0
    # game.send_game_command(f"pukename change_num_of_monster_to_spawn 0")

    # Play until the game (episode) is over.
    while not game.is_episode_finished():

        iteration += 1
        # Get the state.
        game_state = game.get_state()

        # Analyze the state.
        action, action_index = agent.choose_action(game_state)
        
        # Make your action.
        game.make_action(action)
        if game.is_episode_finished():
            break

        next_game_state = game.get_state()
        done = game.is_player_dead()

        ### TRAIN YOUR AGENT HERE
        if train:
            agent.agent_train(game_state, action, next_game_state, done)

        # Check if player is dead
        if game.is_player_dead():
            print("Player died.")
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

    print("Episode finished.")
    print("************************")

    print("Results:")
    server_state = game.get_server_state()
    for i in range(len(server_state.players_in_game)):
        if server_state.players_in_game[i]:
            print(
                server_state.players_names[i]
                + ": "
                + str(server_state.players_frags[i])
            )
    print("************************")

    # Starts a new episode. All players have to call new_episode() in multiplayer mode.
    game.new_episode()
    agent.save_weights("model.pth")
    
game.close()


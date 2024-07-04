''' Host is a spectator to whom clients (players) are connected.
Host will wait for player's machine to connect using the -join parameter and then start the game.
After chosen amount of time, host will end the game and present results.
'''

from argparse import ArgumentParser
from time import time

import vizdoom as vzd


parser = ArgumentParser("ViZDoom training on deathmatch map.")
parser.add_argument("-m", "--map", default="map01", help="Use map01 or map03.")
parser.add_argument("-p", "--players", default=4, type=int, help="Number of players.")
parser.add_argument("-d", "--duration", default=120, type=int, help="Number of seconds.")
args = parser.parse_args()
map_name = args.map
players = args.players
duration = args.duration

if map_name not in ["map01", "map03"]:
    raise ValueError("Please use map01 or map03.")

game = vzd.DoomGame()
game.load_config("scenarios/deathmatch.cfg")
game.set_doom_map(map_name)

game.add_game_args(
    f"-host {1 + players} " # host + clients
    "-port 5029 "
    f"{'' if map_name[-1] == '1' else '-deathmatch '}"  # Deathmatch mode does not apply on map01
    "+viz_connect_timeout 5 "
    "+sv_forcerespawn 1 " # Can respawn
    "+sv_noautoaim 1 " # Disable autoaim for all players
    "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
    "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
    "+sv_nocrouch 1 "  # Disables crouching.
    "+viz_respawn_delay 2 "  # Sets delay between respawns (in seconds, default is 0).
    "+viz_spectator 1 "
    "+freelook 1"
)

game.add_game_args("+name HostSpectator +colorset 4")
game.set_screen_resolution(vzd.ScreenResolution.RES_1920X1080)
game.set_mode(vzd.Mode.SPECTATOR)
game.set_console_enabled(True)
game.init()

start_time = time()
while not game.is_episode_finished():
    game.advance_action()

    if time() - start_time > duration:
        break

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

game.close()

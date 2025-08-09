# A minimal file to satisfy the imports in the code you provided.
# You will likely have to adjust these values to your specific system and game.

is_linux = False  # Set to True if you are on Linux
inputs = [
    {"left": False, "right": False, "accelerate": True, "brake": False},
    {"left": True, "right": False, "accelerate": True, "brake": False},
    {"left": False, "right": True, "accelerate": True, "brake": False},
    {"left": False, "right": False, "accelerate": False, "brake": False},
]
action_forward_idx = 0
update_inference_network_every_n_actions = 100
tmi_protection_timeout_s = 200
timeout_during_run_ms = 1000
timeout_between_runs_ms = 1000
game_reboot_interval = 3000
W_downsized = 160
H_downsized = 90
n_zone_centers_extrapolate_before_start_of_map = 1
n_zone_centers_extrapolate_after_end_of_map = 1
max_allowable_distance_to_virtual_checkpoint = 100
n_contact_material_physics_behavior_types = 1
margin_to_announce_finish_meters = 1000
#!/bin/bash
export CARLA_ROOT=/home/eidos/Workspace/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True

# TCP evaluation
# export ROUTES=leaderboard/data/evaluation_routes/routes_lav_valid_1_route_town02_00.xml
export ROUTES=leaderboard/data/evaluation_routes/routes_lav_valid_town05_second.xml
# export ROUTES=leaderboard/data/evaluation_routes/routes_lav_valid_town05_first.xml
# export ROUTES=leaderboard/data/evaluation_routes/routes_lav_valid.xml
export TEAM_AGENT=team_code/tcp_agent.py
export TEAM_CONFIG=/home/eidos/Warehouse/Teledriving/TCP_agent/epoch=59-last.ckpt
export CHECKPOINT_ENDPOINT=results_TCP.json
export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json
# export SAVE_PATH=/home/eidos/Warehouse/Teledriving/TCP_eval_result/results_TCP

# export CUDA_VISIBLE_DEVICES=-1
# JPEG | J2K | BPG | JSCC | AE
export MODEL_TYPE=J2K
# For JPEG and J2K, higher is better. For BPG lower is better.
export QUALITY=70
# Only for JSCC
export PATH_VAE_MODEL=/home/eidos/Warehouse/Teledriving/model_Infocom2024/VAE/ae_noNorm_snr0/ae_model_epoch_30_iter_355380.pth
# Log
export USE_WANDB=False
# Gym
# export FIFO_PATH=/home/eidos/Workspace/GitKraken_ws/meta_driving/fifo_space

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}



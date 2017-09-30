#!/usr/bin/env bash
# Purpose: create a tmux session for the fast-ai project

SESSION_NAME="fast-ai"
FAST_AI_DIR="$HOME/host-work/fast-ai"

# create the session and the first window
tmux new-session -s ${SESSION_NAME} -n home -d
tmux send-keys -t ${SESSION_NAME}:1 "cd $HOME" C-m

# create the desktop window
tmux new-window -n desktop -t ${SESSION_NAME}:2
tmux send-keys -t ${SESSION_NAME}:2 "ssh desktop" C-m
tmux send-keys -t ${SESSION_NAME}:2 "echo -e \"\e[31m\" LOGGED INTO (hostname)" C-m

# create the host fast-ai window
tmux new-window -n fast-ai -t ${SESSION_NAME}:3
tmux send-keys -t ${SESSION_NAME}:3 "cd ${FAST_AI_DIR}" C-m
tmux send-keys -t ${SESSION_NAME}:3 "pwd" C-m

# start out in the fast-ai window
tmux select-window -t ${SESSION_NAME}:3

# attach to session
tmux attach-session -t ${SESSION_NAME}
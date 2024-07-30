import random
import numpy as np
import tensorflow as tf
from Models import Actor, DuelingActorCritic
import os
import json

distances_eval = [2 for _ in range(5)] + [4 for _ in range(5)] + [6 for _ in range(5)] + [8 for _ in range(5)] + [10 for _ in range(5)] + [15 for _ in range(5)]

def createsStates(state, k=3):
    states = []
    goal = state.copy()
    for i in range(len(distances_eval)):
        state = goal.copy()
        for _ in range(distances_eval[i]):
            action_index = random.randint(0, 2)
            neighbors_states = [np.roll(state, 1), np.roll(state, -1), np.concatenate([state[:k][::-1], state[k:]])]
            state = neighbors_states[action_index]

        if np.all(state == goal):
            action_index = random.randint(0, 2)
            neighbors_states = [np.roll(state, 1), np.roll(state, -1), np.concatenate([state[:k][::-1], state[k:]])]
            state = neighbors_states[action_index]

        states.append(state)
    return states

def get_neighbors_topspin(state):
    return [list(np.roll(state.copy(), 1)),
            list(np.roll(state.copy(), -1)),
            list(np.concatenate([state.copy()[:3][::-1], state.copy()[3:]]))]

def eval(model, states, goal, modelType):
    rewards = {}
    for i in range(len(distances_eval)):
        state = states[i]
        rewards[i] = 0
        for t in range(100):
            if np.all(state == goal):
                done = True
            else:
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                state_tensor = tf.expand_dims(state_tensor, 0)  # Add batch dimension

                if modelType == 0:
                    action_probs = model(state_tensor)
                else:
                    action_probs, _ = model(state_tensor)

                action_probs = tf.squeeze(action_probs)  # Ensure correct shape
                action = np.random.choice(3, p=action_probs.numpy())

                next_state = get_neighbors_topspin(state)[action]
                done = np.all(state == goal)
                if done:
                    break
                else:
                    rewards[i] += 1

                state = next_state

            if done:
                break

    return rewards


def load_model_weights(modelType, n, rewardType, distance):
    model_folder = f'ModelType_{modelType}/N_{n}/RewardType_{rewardType}/Distance_{distance}'
    if modelType == 0:
        model_file = f'actor_model_{n}_{rewardType}_{distance}.h5'
        model_path = os.path.join(model_folder, model_file)
        dummy_input = np.zeros((1, n))
        model = Actor()
        model(dummy_input)
        model.load_weights(model_path, skip_mismatch=False)
    elif modelType == 1:
        model_file = f'dueling_actor_critic_model_{n}_{rewardType}_{distance}.h5'
        model_path = os.path.join(model_folder, model_file)
        dummy_input = np.zeros((1, n))
        model = DuelingActorCritic()
        model(dummy_input)
        model.load_weights(model_path, skip_mismatch=False)

    return model


def evaluate():
    ns = [5, 7]
    rewardTypes = [0, 1, 2, 3]
    distances = [6, 8, 10]
    modelTypes = [0, 1]
    for n in ns:
        goal = np.arange(1, n + 1)
        states = createsStates(goal)
        indexes = {}
        for idx in range(len(states)):
            indexes[idx] = str(list(states[idx]))
        with open(os.path.join("Files", f'indexes_{n}.json'), 'w') as file:
            json.dump(indexes, file)

        for modelType in modelTypes:
            for rewardType in rewardTypes:
                for distance in distances:
                    try:
                        model = load_model_weights(modelType, n, rewardType, distance)
                        rewards = eval(model, states, goal, modelType)
                        with open(os.path.join("Files", f'file_{n}_{rewardType}_{distance}_{modelType}.json'), 'w') as file:
                            json.dump(rewards, file)

                    except Exception as e:
                        print(f"Error evaluating ModelType {modelType}, RewardType {rewardType}, Distance {distance}: {e}")
                        continue



def evaluateOnFly(model, states, goal, n, rewardType, distance, modelType):
    try:
        rewards = eval(model, states, goal, modelType)
        with open(os.path.join("Files", f'file_{n}_{rewardType}_{distance}_{modelType}.json'), 'w') as file:
            json.dump(rewards, file)

    except Exception as e:
        print(f"Error evaluating ModelType {modelType}, RewardType {rewardType}, Distance {distance}: {e}")
from Bootstrapping import Bootstrapping
from TopSpin import TopSpinEnv
from Models import Actor, Critic, DuelingActorCritic, RewardModel
import TrainingActorCritic as TAC
import TrainingDulingActorCritic as TDAC
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import tensorflow as tf
import os
from Evaluation import createsStates, evaluateOnFly, evaluate


def run_actor_critic(env, save_dir, n, rewardType, distance, goal):
    actor = Actor()
    critic = Critic()
    optimizer_actor = Adam()
    optimizer_critic = Adam()

    lr_schedule_actor = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=500,
        decay_rate=0.97)
    optimizer_actor.learning_rate = lr_schedule_actor

    lr_schedule_critic = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=500,
        decay_rate=0.97)
    optimizer_critic.learning_rate = lr_schedule_critic

    rewards, actor_losses, critic_losses = TAC.train_model(env, actor, critic, optimizer_actor, optimizer_critic, goal,
                                                           0.97, 200, 100)
    actor.save_weights(os.path.join(save_dir, f'actor_model_{n}_{rewardType}_{distance}.weights.h5'))
    critic.save_weights(os.path.join(save_dir, f'critic_model_{n}_{rewardType}_{distance}.weights.h5'))
    return actor


def run_dueling_actor_critic(env, save_dir, n, rewardType, distance, goal):
    model = DuelingActorCritic()
    optimizer = Adam()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=500,
        decay_rate=0.97)
    optimizer.learning_rate = lr_schedule

    rewards, losses = TDAC.train_model(env, model, optimizer, goal, 0.97, 200, 100)
    model.save_weights(os.path.join(save_dir, f'dueling_actor_critic_model_{n}_{rewardType}_{distance}.weights.h5'))
    return model


def train_reward_heuristic(bootstrapping):
    rewardOptimizer = Adam()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=500,
        decay_rate=0.97)
    rewardOptimizer.learning_rate = lr_schedule
    bootstrapping.set(RewardModel(), rewardOptimizer)
    return bootstrapping.bootstrappingTraining()


def run_main():
    ns = [5, 7]
    k = 3
    rewardTypes = [4]
    distances = [6, 8, 10]
    modelTypes = [0, 1]
    for n in ns:
        goal = np.arange(1, n + 1)
        states = createsStates(goal)
        bootsrapping = Bootstrapping(n, k)
        bootsrapping_heurisitc = train_reward_heuristic(bootsrapping)
        # indexes = {}
        # for idx in range(len(states)):
        #     indexes[idx] = str(list(states[idx]))
        # with open(os.path.join("Files", f'indexes_{n}.json'), 'w') as file:
        #     json.dump(indexes, file)

        for modelType in modelTypes:
            model_folder = f'ModelType_{modelType}'
            os.makedirs(model_folder, exist_ok=True)

            for rewardType in rewardTypes:
                reward_folder = os.path.join(model_folder, f'N_{n}', f'RewardType_{rewardType}')
                os.makedirs(reward_folder, exist_ok=True)

                rewardNet = None
                if rewardType == 3:
                    rewardNet = bootsrapping_heurisitc

                for distance in distances:
                    env = TopSpinEnv(n=n, k=k, distance=distance, rewardType=rewardType, rewardNet=rewardNet)
                    save_dir = os.path.join(reward_folder, f'Distance_{distance}')
                    os.makedirs(save_dir, exist_ok=True)

                    if modelType == 0:
                        model = run_actor_critic(env, save_dir, n, rewardType, distance, goal)
                        evaluateOnFly(model, states, goal, n, rewardType, distance, modelType)
                    elif modelType == 1:
                        model = run_dueling_actor_critic(env, save_dir, n, rewardType, distance, goal)
                        evaluateOnFly(model, states, goal, n, rewardType, distance, modelType)


if __name__ == '__main__':
    run_main()
    # evaluate()

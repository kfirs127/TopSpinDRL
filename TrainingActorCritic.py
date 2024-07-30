import tensorflow as tf
import numpy as np
import random


def optimize_batch(actor, critic, optimizer_actor, optimizer_critic, states, actions, rewards, next_states, dones,
                   gamma):
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # Convert dones to float

    with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
        critic_values = tf.squeeze(critic(states, training=True))
        critic_values_next = tf.squeeze(critic(next_states, training=True))

        td_targets = rewards + gamma * critic_values_next * (1 - dones)
        td_errors = td_targets - critic_values

        # Critic loss
        critic_loss = tf.reduce_mean(td_errors ** 2)

        # Actor loss
        action_probs = actor(states, training=True)
        log_action_probs = tf.math.log(
            tf.reduce_sum(action_probs * tf.one_hot(actions, action_probs.shape[-1]), axis=1) + 1e-10)
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
        actor_loss = -tf.reduce_mean(log_action_probs * td_errors + 0.01 * entropy)

        if not tf.math.is_nan(actor_loss) and not tf.math.is_inf(actor_loss):
            actor_grads = tape_actor.gradient(actor_loss, actor.trainable_variables)
            optimizer_actor.apply_gradients(zip(actor_grads, actor.trainable_variables))

        if not tf.math.is_nan(actor_loss) and not tf.math.is_inf(actor_loss):
            critic_grads = tape_critic.gradient(critic_loss, critic.trainable_variables)
            optimizer_critic.apply_gradients(zip(critic_grads, critic.trainable_variables))

    return actor_loss, critic_loss


def train_model(env, actor, critic, optimizer_actor, optimizer_critic, goal, gamma, max_episodes, max_steps):
    episode_Rewards = []
    actor_losses = []
    critic_losses = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        print(f'state:{state}')
        episode_reward = 0

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        done = 0
        for t in range(max_steps):
            if np.all(state == goal):
                done = 1
            else:
                state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
                state_tensor = tf.expand_dims(state_tensor, 0)

                action_probs = actor(state_tensor)
                if action_probs is None or tf.reduce_any(tf.math.is_nan(action_probs)):
                    action = random.randint(0, 2)
                else:
                    action_probs = tf.squeeze(action_probs)
                    action = np.random.choice(env.action_space.n, p=action_probs.numpy())

                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state

            if done == 1:
                break

        if done == 1 and len(states) > 0:
            actor_loss, critic_loss = optimize_batch(actor, critic, optimizer_actor, optimizer_critic, states, actions,
                                                     rewards, next_states, dones, gamma)
            actor_losses.append(actor_loss.numpy())
            critic_losses.append(critic_loss.numpy())
            episode_Rewards.append(episode_reward)
            print(
                f"Episode {episode + 1}: Reward: {episode_reward}, Actor Loss: {actor_loss:.2f}, Critic Loss: {critic_loss:.2f}")
        else:
            episode_Rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward: {episode_reward}")

    return episode_Rewards, actor_losses, critic_losses
import tensorflow as tf
import numpy as np
import random


def optimize_batch(actor_critic, optimizer, states, actions, rewards, next_states, dones, gamma):
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # Convert dones to float

    with tf.GradientTape() as tape:
        action_probs, critic_values = actor_critic(states, training=True)
        _, critic_values_next = actor_critic(next_states, training=True)

        td_targets = rewards + gamma * tf.squeeze(critic_values_next) * (1 - dones)
        td_errors = td_targets - tf.squeeze(critic_values)

        # Critic loss
        critic_loss = tf.reduce_mean(td_errors ** 2)

        # Actor loss
        log_action_probs = tf.reduce_sum(
            tf.one_hot(actions, action_probs.shape[-1]) * tf.math.log(action_probs + 1e-10), axis=1
        )
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
        actor_loss = -tf.reduce_mean(log_action_probs * td_errors + 0.01 * entropy)

        # Combined loss
        total_loss = actor_loss + critic_loss

    grads = tape.gradient(total_loss, actor_critic.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor_critic.trainable_variables))

    return total_loss


def train_model(env, model, optimizer, goal, gamma, max_episodes, max_steps):
    episode_Rewards = []
    episode_Losses = []

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

                action_probs, _ = model(state_tensor)
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
            episode_loss = optimize_batch(model, optimizer, states, actions, rewards, next_states, dones, gamma)
            episode_Losses.append(episode_loss.numpy())
            episode_Rewards.append(episode_reward)
            print(
                f"Episode {episode + 1}: Reward: {episode_reward}, Loss: {episode_loss:.2f}")
        else:
            episode_Rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward: {episode_reward}")

    return episode_Rewards, episode_Losses

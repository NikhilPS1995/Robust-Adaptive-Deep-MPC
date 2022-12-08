import numpy as np


class MPCPolicyBaseline():
    
    def __init__(self,
                 ac_dim,
                 ob_dim, 
                 ac_space_low, 
                 ac_space_high, 
                 dyn_models,
                 horizon,
                 N,
                 goal,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.goal = goal

        self.ob_dim = ob_dim

        # action space
        self.ac_dim = ac_dim
        self.low = ac_space_low
        self.high = ac_space_high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            # Uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range [self.low, self.high]
            random_action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
            return random_action_sequences # (N, H, D_action)
        elif self.sample_strategy == 'cem':
            # Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (For the first iteration, we instead sample uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                # - Update the elite mean and variance
                if i == 0:
                    sampled_action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
                else:
                    sampled_action_sequences = np.random.normal(loc=running_mean, scale=np.sqrt(running_var), size=(num_sequences, horizon, self.ac_dim))
                all_rewards = self.evaluate_candidate_sequences(sampled_action_sequences, obs)
                elites_indices = all_rewards.argsort()[-self.cem_num_elites:]
                if i == 0:
                    running_mean = np.mean(sampled_action_sequences[elites_indices, :, :], axis=0) # The mean matrix is of shape (horizon, self.ac_dim)
                    running_var = np.var(sampled_action_sequences[elites_indices, :, :], axis=0) # The var matrix is of shape (horizon, self.ac_dim)
                else:
                    running_mean = self.cem_alpha * np.mean(sampled_action_sequences[elites_indices, :, :], axis=0) + (1-self.cem_alpha) * running_mean
                    running_var = self.cem_alpha * np.var(sampled_action_sequences[elites_indices, :, :], axis=0) + (1-self.cem_alpha) * running_var
            # Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self.ac_dim)
            # best_action_index = all_rewards.argsort()[-1]
            # cem_action = sampled_action_sequences[best_action_index]
            cem_action = np.mean(sampled_action_sequences[elites_indices, :, :], axis=0)
        
            return cem_action[None] # add an axis to the first dimension
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # For each model in ensemble, compute the predicted sum of rewards for each candidate action sequence.
        # Then, return the mean predictions across all ensembles. The return value should be an array of shape (N,)
        num_sequences = candidate_action_sequences.shape[0]
        rewards_action_sequences_running_sum_for_ensemble = np.zeros((num_sequences,))
        for model in self.dyn_models:
            rewards_action_sequences_running_sum_for_ensemble += self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)

        return rewards_action_sequences_running_sum_for_ensemble / num_sequences

    def get_action(self, obs):
        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon, obs=obs)
        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            action_to_take = candidate_action_sequences[0]#[0][None]
            predicted_obs = []
            for k in range(self.horizon):
                obs = obs[None, ...] if len(obs.shape) < 2 else obs.reshape(1, 2)
                obs = self.dyn_models[0].get_prediction(obs, action_to_take[k].reshape(1, 1))
                predicted_obs.append(obs[0])
            predicted_obs = np.array(predicted_obs)
            return action_to_take, predicted_obs
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)] 
            action_to_take = best_action_sequence#[0]
            # a[None] is equivalent to a[None, :, :] or a[None, ...] or a[np.newaxis, :, :] (add an axis in the first dimension)
            # action_to_take = action_to_take[None]  # Unsqueeze the first index 
            predicted_obs = []
            for k in range(self.horizon):
                obs = obs[None, ...] if len(obs.shape) < 2 else obs.reshape(1, 2)
                obs = self.dyn_models[0].get_prediction(obs, action_to_take[k].reshape(1, 1))
                predicted_obs.append(obs[0])
            predicted_obs = np.array(predicted_obs)
            return action_to_take, predicted_obs

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """
        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        N = candidate_action_sequences.shape[0]
        sum_of_rewards = np.zeros(N)  
        # For each candidate action sequence, predict a sequence of states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        
        assert candidate_action_sequences.shape[1] == self.horizon
        obs = np.tile(obs, (N, 1)) # repeat the current observation vertically
        for t in range(self.horizon):
            actions = candidate_action_sequences[:, t, :]
            rewards, dones = self.get_reward(obs, actions)
            sum_of_rewards += rewards
            obs = model.get_prediction(obs, actions)
        return sum_of_rewards

    def get_reward(self, obs, actions):
        """
        obs: N x D_obs
        self.goal: D_obs
        """
        return -np.sum((self.goal - obs) ** 2, axis=1), False
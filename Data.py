from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Conv1D, Activation, Dropout, MaxPooling1D, Flatten
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np
from pathlib import Path
from shutil import rmtree
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 5000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 100 # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Update Target model after x steps

# Exploration settings
epsilon = 1
EPSILON_DECAY = 0.999975
MIN_EPSILON = 0.001

# Model settings
NB_ACTIONS = 96
SAVE_INTERVAL = 5


class TermENV:

    def __init__(self):
        self.range = .48
        self.NB_ACTIONS = 96
        self.array_len = 1000
        self.length_penalty = .5
        self.learning_reward = 1
        self.variety_reward = 1
        self.blank_penalty = 10
        self.reset()

    def step(self, action):
        enc_ascii = action + 32
        if enc_ascii != 127:
            self.cmd += chr(enc_ascii)
            self.cmd_in = False
        else:
            self.cmd_in = True
        if self.cmd_in:
            self.reward = 0
            if not self.cmd:
                self.reward -= self.blank_penalty
            proc = Popen(self.cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
            self.cmd = ''
            try:
                stdout = proc.communicate(timeout=5)[0].decode()
                exitcode = proc.returncode
            except TimeoutExpired:
                proc.kill()
                stdout = proc.communicate()[0].decode()
                exitcode = proc.returncode
            self.term_out = ''.join(char for char in stdout if char.isprintable())
            input_data = self.term_out + ' ' + str(Path.cwd()) + '> '
            filename = Path('mem.txt')
            filename.touch(exist_ok=True)
            if exitcode == 0:
                with open('mem.txt', 'r+') as mem:
                    for line in stdout.splitlines():
                        if line + '\n' not in mem:
                            mem.write(line + '\n')
                            self.reward += self.learning_reward
            print('\n')
            print(stdout)
            print(str(Path.cwd()) + '> ', end='', flush=True)
        else:
            input_data = self.term_out + ' ' + str(Path.cwd()) + '> ' + self.cmd
            print(input_data[-1], end='', flush=True)
            if self.prev_cmd:
                if self.cmd[-1] not in self.prev_cmd:
                    self.reward += self.variety_reward
            self.prev_cmd = self.cmd
            self.reward -= self.length_penalty
        idxs = np.swapaxes((np.atleast_2d((np.frombuffer(input_data.encode(), dtype=np.uint8) - 31) / 100)), 0, 1)
        if idxs.shape[0] < self.array_len:
            self.observation = np.append(idxs, np.zeros(((self.array_len - idxs.shape[0]), 1)), axis=0)
        else:
            self.observation = np.resize(idxs, (1, self.array_len))
        return self.observation, self.reward, self.cmd_in

    def reset(self):
        idxs = np.swapaxes((np.atleast_2d((np.frombuffer((str(Path.cwd()) + '> ').encode(), dtype=np.uint8) - 31) / 100)), 0, 1)
        if idxs.shape[0] < self.array_len:
            self.observation = np.append(idxs, np.zeros(((self.array_len - idxs.shape[0]), 1)), axis=0)
        else:
            self.observation = np.resize(idxs, (1, self.array_len))
        self.reward = 0
        self.term_out = ''
        self.prev_cmd = ''
        self.cmd = ''
        return self.observation


env = TermENV()


class DQNAgent:
    def __init__(self):

        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv1D(256, 3, input_shape=env.observation.shape))
        model.add(Activation("relu"))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        model.add(Conv1D(256, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(NB_ACTIONS, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]


agent = DQNAgent()
if Path("SavedModels/").is_dir():
    agent.model = load_model('SavedModel/Model.keras')
current_state = env.reset()
save = 0
while True:
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(agent.get_qs(current_state))
    else:
        # Get random action
        action = np.random.randint(0, env.NB_ACTIONS)
    new_state, reward, done = env.step(action)

    agent.update_replay_memory((current_state, action, reward, new_state, done))
    agent.train(done)
    current_state = new_state
    if done:
        save += 1
    if save > SAVE_INTERVAL:
        if Path('SavedModel/').is_dir():
            rmtree('SavedModel/')
        Path('SavedModel/').mkdir(parents=True, exist_ok=True)
        save_model(agent.model, 'SavedModel/Model.keras')
        save = 0
    if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

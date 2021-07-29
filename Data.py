from keras.models import Sequential, load_model, save_model
from keras.layers import GRU, Dense
from pathlib import Path
from shutil import rmtree
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import numpy as np
import tensorflow as tf
import random
from gc import collect
from keras.backend import clear_session
from datetime import datetime

tf.get_logger().setLevel('ERROR')

# env adjustments
cmd = 'echo Hello World!'
length_penalty = .25
learning_reward = 1
variety_reward = 5
max_cmd = 100
blank_penalty = max_cmd * length_penalty

# model adjustments
hidden_layers = 32
layer_neurons = 128
nb_actions = 96
model_num = 0

# training adjustments
total_models = 50
starting_fitness = 0
# maximum and minimum percentage mutated
mutation_max = 50
mutation_min = 10
mutation_value = nb_actions / 100

# variable assignment
new_weights = []
aux_weights = []
main_pool = []
fitness = []
aux_pool = []
aux_parent1 = 0
aux_parent2 = 0
fubar_pool = []
fubar_parent1 = 0
fubar_parent2 = 0
fubar_fallback = 2

init = True
cmd_in = True
highest_fitness = -(max_cmd * length_penalty)
term_out = ''
prev_cmd = ''
error_count = 0
update_count = 0
no_update = 0
global e
mutation_max = round(1 - (mutation_max / 100), 2)
mutation_min = round(1 - (mutation_min / 100), 2)
mutation_rate = mutation_min


def term_interact():
    global term_out, cmd, prev_cmd
    if cmd_in:
        term_out = ''
        prev_cmd = ''
        if not cmd:
            fitness[model_num] -= blank_penalty
        proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
        cmd = ''
        try:
            stdout = proc.communicate(timeout=5)[0].decode()
            exitcode = proc.returncode
        except TimeoutExpired:
            proc.kill()
            stdout = proc.communicate()[0].decode()
            exitcode = proc.returncode
        term_out = ''.join(char for char in stdout if char.isprintable())
        input_data = term_out + ' ' + str(Path.cwd()) + '> '
        filename = Path('mem.txt')
        filename.touch(exist_ok=True)
        if not init:
            if exitcode == 0:
                with open('mem.txt', 'r+') as mem:
                    for line in stdout.splitlines():
                        if line + '\n' not in mem:
                            mem.write(line + '\n')
                            fitness[model_num] += learning_reward
        print('\n')
        print(stdout)
        print(str(Path.cwd()) + '> ', end='', flush=True)
    else:
        input_data = term_out + ' ' + str(Path.cwd()) + '> ' + cmd
        print(input_data[-1], end='', flush=True)
        if prev_cmd:
            if cmd[-1] not in prev_cmd:
                fitness[model_num] += variety_reward
        prev_cmd = cmd
        if not init:
            fitness[model_num] -= length_penalty
    neural_input = np.atleast_3d((np.frombuffer(input_data.encode(), dtype=np.uint8) - 31) / 100)
    return neural_input


def create_model():
    model = Sequential()
    for layer in range(hidden_layers):
        model.add(GRU(layer_neurons, name='GRU' + str(layer), return_sequences=True))
    model.add(GRU(layer_neurons, name='GRU' + str(hidden_layers)))
    model.add(Dense(nb_actions, name='output', activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    return model


def model_mutate(weights):
    global mutation_rate
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.uniform(0, 1) > mutation_rate:
                change = random.uniform(-mutation_value, mutation_value)
                weights[i][j] += change
    return weights


def model_crossover(pool, parent_x1, parent_x2):

    weight1 = pool[parent_x1].get_weights()
    weight2 = pool[parent_x2].get_weights()

    new_weight1 = weight1
    new_weight2 = weight2
    for i in range(len(new_weight1)):
        if random.uniform(0, 1) > .90:
            gene = random.randint(0, len(new_weight1) - 1)
            new_weight1[gene] = weight2[gene]
            new_weight2[gene] = weight1[gene]
    return np.asarray([new_weight1, new_weight2])


def cleanup():
    global mutated1, mutated2, new_weights, parent1, parent2, cross_over_weights, prediction, action, enc_ascii
    if 'mutated1' in globals():
        del mutated1
    if 'mutated2' in globals():
        del mutated2
    if 'parent1' in globals():
        del parent1
    if 'parent2' in globals():
        del parent2
    if 'cross_over_weights' in globals():
        del cross_over_weights

    del new_weights, prediction, action, enc_ascii
    new_weights = []
    clear_session()
    collect()


def save_pool():
    if Path("SavedModels/").is_dir():
        rmtree("SavedModels/")
    Path("SavedModels/").mkdir(parents=True, exist_ok=True)
    for xi in range(total_models):
        save_model(main_pool[xi], "SavedModels/model_new" + str(xi) + ".keras")


while True:
    try:
        if Path("SavedModels/").is_dir():
            for i in range(total_models):
                main_pool.append(load_model("SavedModels/model_new" + str(i) + ".keras"))
                fitness.append(starting_fitness)
        else:
            for i in range(total_models):
                model = create_model()
                fitness.append(starting_fitness)
                main_pool.append(model)

        while True:
            while model_num < total_models:
                prediction = main_pool[model_num].predict(term_interact(), batch_size=1)
                init = False
                action = np.argmax(prediction)
                enc_ascii = action + 32
                if len(cmd) < max_cmd:
                    if enc_ascii != 127:
                        cmd += chr(enc_ascii)
                        cmd_in = False
                        continue
                    else:
                        cmd_in = True
                        model_num += 1
                        continue
                else:
                    cmd_in = True
                    model_num += 1
                    continue
            model_num = 0

            updated = False
            for select in range(total_models):
                if fitness[select] >= highest_fitness:
                    updated = True
                    highest_fitness = fitness[select]

            parent1 = random.randint(0, total_models - 1)
            parent2 = random.randint(0, total_models - 1)
            for i in range(total_models):
                if fitness[i] >= fitness[parent1]:
                    parent1 = i
            for j in range(total_models):
                if j != parent1:
                    if fitness[j] >= fitness[parent2]:
                        parent2 = j

            if updated:
                aux_pool = main_pool
                aux_parent1 = parent1
                aux_parent2 = parent2
                mutation_rate = mutation_min
                update_count += 1
                if update_count == 2:
                    fubar_pool = main_pool
                    fubar_parent1 = parent1
                    fubar_parent2 = parent2
                    update_count = 0

            else:
                if mutation_rate > mutation_max:
                    mutation_rate -= .01

            for select in range(total_models // 2):
                if updated:
                    cross_over_weights = model_crossover(main_pool, parent1, parent2)
                else:
                    no_update += 1
                    if no_update != fubar_fallback:
                        cross_over_weights = model_crossover(aux_pool, aux_parent1, aux_parent2)
                    else:
                        if not fubar_pool:
                            fubar_pool = aux_pool
                            fubar_parent1 = aux_parent1
                            fubar_parent2 = aux_parent2
                        cross_over_weights = model_crossover(fubar_pool, fubar_parent1, fubar_parent2)
                        no_update = 0
                mutated1 = model_mutate(cross_over_weights[0])
                mutated2 = model_mutate(cross_over_weights[1])
                new_weights.append(mutated1)
                new_weights.append(mutated2)

            for reset in range(total_models):
                fitness[reset] = starting_fitness
            for select in range(len(new_weights)):
                main_pool[select].set_weights(new_weights[select])
            cleanup()
            save_pool()

    except Exception as e:
        logfile = Path('error_log.txt')
        logfile.touch(exist_ok=True)
        with open("error_log.txt", "a") as log:
            log.write(str(datetime.now()) + ' ' + str(e))
            log.write('\n')
        error_count += 1
        if error_count <= 10:
            continue
        else:
            print(e)
            break

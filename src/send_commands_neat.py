from __future__ import print_function
import os
import neat
import visualize

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey

# 2-input XOR inputs and expected outputs.
# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]
xor_inputs = [[0.0, 0.1, 0.01, 0.11, 0.2]]
xor_outputs = [[0], [0], [1], [0], [0]]


### input is infrared sensors, output is movement of robot ###


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def eval_genomes(genomes, config):
    time_out = 10
    timeout_start = time.time()
    min_distance = 0.3
    left_vs_right = 0

    signal.signal(signal.SIGINT, terminate_program)
    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7") #VU
    rob = robobo.SimulationRobobo().connect(address='192.168.68.114', port=19997)  # Caya's home
    # rob = robobo.SimulationRobobo().connect(address='192.168.2.9', port=19997) # Sebby's home
    # rob = robobo.SimulationRobobo().connect(address='localhost', port=19997)

    rob.play_simulation()

    # Following code moves the robot
    # for i in range(2):
    #        print("robobo is at {}".format(rob.position()))
    #        rob.move(20, 20, 1000)

    print("robobo is at {}".format(rob.position()))

    # Following code gets an image from the camera
    image = rob.get_image_front()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    cv2.imwrite("test_pictures.png", image)

    for genome_id, genome in genomes:
        bump = 0
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        while time.time() < timeout_start + time_out:
            moves = net.activate(rob.read_irs()[3:8])
            max_value = max(moves)
            action = moves.index(max_value)
            if action == 0:
                rob.move(50, 50, 10)  # straight
            elif action == 1:
                rob.move(8, 20, 10)  # left
            elif action == 2:
                rob.move(20, 8, 10)  # right
            elif action == 3:
                rob.move(-30, 30, 10)  # sharp left
            elif action == 4:
                rob.move(30, -30, 10)  # sharp right
            else:
                print('index not found')

            bump_identified_per_sensor = [bump + 1 for sensor in rob.read_irs()[3:8] if
                                          sensor != False and sensor < min_distance]  # sensor value is a number and below threshold
            left_or_right_action = [left_vs_right + 1 if action in [1, 2, 3, 4] else left_vs_right]

            bump = [1 if len(bump_identified_per_sensor) != 0 and max(bump_identified_per_sensor) == 1 else 0]

            # penalty for every time it bumps or turns left or right
            genome.fitness -= bump[0] * -50 - left_or_right_action[0] * -2
            print(genome.fitness)
            bump_identified_per_sensor = []
            left_or_right_action = []

        time_out = 10
        timeout_start = time.time()

    rob.pause_simulation()
    rob.stop_world()
    rob.disconnect()

    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = net.activate(xi)
    #     genome.fitness -= (output[0] - xo[0]) ** 2


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10)

    # # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    #
    # # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    #
    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # # visualize.draw_net(config, winner, True, node_names=node_names)
    # # visualize.plot_stats(stats, ylog=False, view=True)
    # # visualize.plot_species(stats, view=True)
    #
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)

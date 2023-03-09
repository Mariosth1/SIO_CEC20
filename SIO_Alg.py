#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:16:52 2020

@author: altzanetos
@modifications : M.Thymianis

"""

import numpy as np
from copy import deepcopy

from opfunu.cec.cec2020 import engineering
from opfunu.cec.cec2020.constant import benchmark_function as BF

"""
Objective function 
import it as quality
quality = quality(position)
"""


def quality(position, problem):

    prob = "p" + str(problem)
    fx, gx, hx = getattr(engineering, prob)(*position)

    # Add constraint handling based on paper E-2443 from CEC2020 RW constrained optimization
    if not isinstance(hx, np.ndarray): hx = np.array([hx])
    if not isinstance(gx, np.ndarray): gx = np.array([gx])

    violation_term = np.sum(np.maximum(0, gx)) + np.sum(np.abs(equal) for equal in hx if np.abs(equal) - 1e-4 > 0)
    violation_term = np.divide(violation_term, gx.shape[0] + hx.shape[0])

    # Penalty term
    if violation_term >= 1e3 and violation_term < 1e4:
        penalty_term = violation_term * 0.5e-1
    elif violation_term >= 1e4:
        penalty_term = violation_term * 0.35e-2
    else:
        penalty_term = violation_term

    return fx + penalty_term, violation_term, penalty_term


class Agent:

    def __init__(self, bounds, pwr, refer_intensity, problem):
        self.position = np.random.uniform(bounds[0], bounds[1], size=(1, bounds.shape[1]))
        [self.quality, self.violation, self.penalty] = quality(self.position, problem)  # quality of solution based on objective function
        self.var_ranges = bounds[0] - bounds[1]
        self.r0 = self.effective_radius(self.var_ranges)
        self.intensity = self.intensity_calc(pwr, refer_intensity)

    @staticmethod
    def effective_radius(var_ranges):

        r0 = np.divide(var_ranges, 4)  # Maybe Change this?
        return r0

    def target(self, target_quality):
        if self.quality < target_quality:
            target = self.position.copy()  # best position found by agent
            target_quality = self.quality.copy()

        return target, target_quality

    def intensity_calc(self, pwr, refer_intensity):
        acoustic_power_output = np.exp(self.quality * 10 ** (-pwr))  # WA
        area = 4 * np.pi * self.r0 ** 2
        intensity = 10 * np.log(np.divide(np.divide(acoustic_power_output, area), refer_intensity))

        return intensity

class global_best_solution:
    def __init__(self, p, q, v, pen):
        self.best_position = p
        self.best_quality = q  #float('inf')
        self.best_violation = v
        self.best_penalty = pen
        self.conv_counter = 0

    def update(self, pos, qual, viol, pen):
        if qual < self.best_quality and viol == 0: # < self.best_violation:
            self.conv_counter = 0
            self.best_quality = qual
            self.best_position = pos
            self.best_violation = viol
            self.best_penalty = pen
        else:
            self.conv_counter += 1



class best_agent:
    """
    Class to keep track of the best agent of the population.
    """

    def __init__(self, p, q, r, v, pen):
        self.best_solution = p
        self.best_quality = q
        self.best_violation = v
        self.best_penalty = pen
        self.r0_best = r

    def update(pos, qual, R0, viol, pen):
        # best_solution = pos
        # best_quality = qual
        # r0_best = R0
        return best_agent(pos, qual, R0, viol, pen)


class SIO:
    '''

    '''
    np.random.seed()

    # Physical analogue's parameters
    I0 = 10 ** (-12)  # reference sound intensity

    angleid = np.array([50, 40, 30, 20, 10, 5])
    s = 0.0008
    pwr = 8  # Change needed here

    """Initialization of parameters and population"""

    def __init__(self, fleet, scans, problem):
        self.fleet = fleet
        # problem's
        self.problem = problem
        out = BF(problem)
        self.dimensions = out["D"]
        lb = np.array([out["xmin"]])
        up = np.array([out["xmax"]])

        # lb = lower_bound * np.ones((1, dimensions))
        # up = upper_bound * np.ones((1, dimensions))
        self.bounds = np.concatenate((lb, up), axis=0)
        self.accept_range = np.subtract(up, lb)
        self.weighted = np.divide(self.accept_range, np.sum(self.accept_range))
        self.population = [Agent(self.bounds, self.pwr, self.I0, self.problem) for _ in range(fleet)]  # population initialization
        self.scans = scans
        self.checkpoint = scans * 0.01

        self.targets = self.population.copy()  # best solution of each agent (initialization)

        # Save the best of all generations


    # def best_agent_attr(self, agent_id):
    # Define best agent's attributes
    #    best_agent.best_solution = self.population[agent_id].position
    #    best_agent.best_quality = self.population[agent_id].quality
    #    best_agent.r0_best = self.population[agent_id].r0
    #    return best_agent



    """Relocation mechanism"""

    def relocation(self, agent_quality, mean_quality, BestAgent):

        if agent_quality > mean_quality:
            relocated_agent = Agent(self.bounds, self.pwr, self.I0, self.problem).position
            r0 = Agent.effective_radius(BestAgent.best_solution)

        else:
            r0 = np.multiply(BestAgent.r0_best, np.random.rand(1, len(BestAgent.r0_best)))
            relocated_agent = BestAgent.best_solution + np.multiply(BestAgent.r0_best, np.random.rand(1, len(r0)))

        return relocated_agent, r0

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    """Full Scan Loop"""

    @staticmethod
    def full_scan_loop(agent, max_angle, problem, gen_per):  # , angleid):

        candidate_position = agent.position  # initial point - agent's current position
        new_position = agent.position
        new_quality = agent.quality
        new_violation = agent.violation
        new_penalty = agent.penalty
        angles = np.fix(max_angle * np.random.rand(1, agent.position.shape[1]))  # angle initialization

        while np.all(angles < 360):
            # Change Here
            r = np.multiply(agent.r0, np.random.rand(1, agent.r0.shape[1]))  # random radius
            movement = np.multiply(r, np.cos(angles))
            candidate_position = np.add(candidate_position, movement)  # calculating new position

            # Relocation of Agents
            out = BF(problem)
            lb = np.array([out["xmin"]])
            up = np.array([out["xmax"]])
            for k in range(candidate_position.shape[1]):
                # Relocation in space
                # if candidate_position[0][k] > up[0][k] or candidate_position[0][k] < lb[0][k]:
                #     candidate_position[0][k] = lb[0][k] + (
                #             up[0][k] - lb[0][k]) * np.cos(candidate_position[0][k]) ** 2
                # else: pass

                # Relocation at bounds
                if candidate_position[0][k] > up[0][k]: candidate_position[0][k] = lb[0][k]
                elif candidate_position[0][k] < lb[0][k]: candidate_position[0][k] = lb[0][k]
                else: pass

            [candidate_pos_quality, candidate_pos_violation, candidate_pos_penalty] =\
                quality(candidate_position, problem)

            # update best-so-far candidate position

            if candidate_pos_quality < agent.quality and candidate_pos_violation < agent.violation:
                new_quality = candidate_pos_quality
                new_position = candidate_position
                new_violation = candidate_pos_violation
                new_penalty = candidate_pos_penalty

            angles = np.add(angles, np.fix(max_angle * np.random.rand(1, agent.r0.shape[1])))  # angle update
        return new_position, new_quality, new_violation, new_penalty

    """Main algorithmic procedure"""

    def optimization(self):

        self.diver_array = []
        self.diver_per = None
        exploration = []

        check = np.zeros((np.size(self.population, 0)))
        fitx = []

        for j in range(len(self.population)):
            fitx.append(self.population[j].quality)

        agent_id = fitx.index(min(fitx))
        BestAgent = best_agent.update(self.population[agent_id].position, self.population[agent_id].quality,
                                      self.population[agent_id].r0, self.population[agent_id].violation,
                                      self.population[agent_id].penalty)

        best_of_all = global_best_solution(BestAgent.best_solution, BestAgent.best_quality, BestAgent.best_violation,
                                           BestAgent.best_penalty)

        for i in range(self.scans):

            for j in range(len(self.population)):

                # Better Copy
                current_agent = deepcopy(self.population[j])

                # Relocation mechanism
                if check[j] == self.checkpoint:
                    [self.population[j].position, self.population[j].r0] =\
                        self.relocation(self.population[j].quality, np.mean(fitx), BestAgent)
                    current_agent = deepcopy(self.population[j])
                    # current_agent.params(self.pwr, self.I0)
                    check[j] = 0  # restart counter

                # Correction of effective radius r0
                # k = current_agent.effective_radius[np.where(current_agent.r0 > self.accept_range)]
                current_agent.r0 = np.where(current_agent.r0 > self.accept_range,
                                            self.accept_range + np.random.rand(1, 1), current_agent.r0)

                # Full Scan Loop

                f_max_angle, = np.where(np.argsort(fitx) == j)  # find value of maximum rotation angle
                idx = np.fix((f_max_angle + 1) / (self.fleet / len(self.angleid)) - 1)
                max_angle = self.angleid[idx.astype(int)]
                [candidate, candidate_quality, candidate_violation, candidate_penalty] =\
                    self.full_scan_loop(current_agent, max_angle, self.problem, (i / self.scans))  # , self.angleid)

                # Update solution if quality is better and violation is smaller
                if candidate_quality < current_agent.quality and candidate_violation < current_agent.violation:
                    self.population[j].position = candidate
                    self.population[j].quality = candidate_quality
                    self.population[j].violation = candidate_violation
                    self.population[j].penalty = candidate_penalty
                    fitx[j] = candidate_quality
                    check[j] = 0
                else:
                    check[j] += 1  # counting consecutive steps without improvement

                magnitude = ((candidate_quality - BestAgent.best_quality) / (10 ** self.pwr) + self.s) * self.weighted
                # Update Intensity
                current_agent.intensity = np.multiply(current_agent.intensity, np.exp(magnitude))

            agent_id = fitx.index(min(fitx))
            BestAgent = best_agent.update(self.population[agent_id].position, self.population[agent_id].quality,
                                          self.population[agent_id].r0, self.population[agent_id].violation,
                                          self.population[agent_id].penalty)


            exploration.append(self.diversification(self.bounds[0].shape[0], len(self.population)))

            best_of_all.update(BestAgent.best_solution, BestAgent.best_quality, BestAgent.best_violation,
                               BestAgent.best_penalty)

        exploitation = [1 - exploration[_] for _ in range(len(exploration))]

        return best_of_all.best_quality, best_of_all.best_penalty, best_of_all.best_position, best_of_all.best_violation,\
               np.average(exploration), np.average(exploitation), best_of_all.conv_counter


    def diversification(self, dimensions, particles_pop):
        # Finding the % of exploration
        i = 0  # This loop can be done easier, fix it later
        agents_matrix = []
        for agent in self.population:

            if i == 0: agents_matrix = agent.position

            else: agents_matrix = np.concatenate((agents_matrix, agent.position))

            i = 1

        dim_median = np.median(agents_matrix, axis=0)  # The median of each column(dimension) of the particle matrix

        div_in_iter = np.sum(dim_median - agents_matrix) / (particles_pop * dimensions) # Average of Diversity of all dimensions

        self.diver_array = np.append(self.diver_array, div_in_iter)

        max_diver = np.max((abs(self.diver_array)))

        return abs(div_in_iter) / max_diver
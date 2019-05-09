"""
# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ActionFilter Implementaion.
@author Chao Gao, cgao3@ualberta.ca. 
Change BOMBING_TEST for different pruning options on bomb placing.
TODO: 1) Moving bomb detection 2) other agent movements. 
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import numpy as np
from pommerman import utility
from pommerman import constants
from collections import deque
import copy
import math
from collections import deque

INT_MAX=9999.0
#3 options, simple | simple_ajdacent | lookahead
#BOMBING_TEST='simple'
#BOMBING_TEST='simple_adjacent' 
BOMBING_TEST='lookahead'
NO_KICKING=True

FLAME_LIFE=2

EPSILON=0.001 
#In original pommerman, bomb_life was represted using float number, 
#0 means there is no bomb
#Here we use EPSILON to distinguish between exploding bomb and no bomb

def _opposite_direction(direction):
    if direction == constants.Action.Left:
        return constants.Action.Right
    if direction == constants.Action.Right:
        return constants.Action.Left
    if direction == constants.Action.Up:
        return constants.Action.Down
    if direction == constants.Action.Down:
        return constants.Action.Up
    return None

def move_moving_bombs_to_next_position(prev_obs, obs):
    def is_moving_direction(bomb_pos, direction):
        rev_d=_opposite_direction(direction)
        rev_pos=utility.get_next_position(bomb_pos, rev_d)
        if not utility.position_on_board(prev_obs['board'], rev_pos):
            return False
        if prev_obs['bomb_life'][rev_pos] - 1 == obs['bomb_life'][bomb_pos] \
            and prev_obs['bomb_blast_strength'][rev_pos] == obs['bomb_blast_strength'][bomb_pos] \
                and utility.position_is_passage(prev_obs['board'], bomb_pos):
            return True
        return False

    bombs=zip(*np.where(obs['bomb_life']>1))
    moving_bombs=[]
    for bomb_pos in bombs:
        moving_dir=None
        for d in [constants.Action.Left, constants.Action.Right, \
            constants.Action.Up, constants.Action.Down]:
            if is_moving_direction(bomb_pos, d):
                moving_dir=d
                break
        if moving_dir is not None:
            moving_bombs.append((bomb_pos, moving_dir))
    board=obs['board']
    bomb_life=obs['bomb_life']
    bomb_blast_strength=obs['bomb_blast_strength']
    for bomb_pos, moving_dir in moving_bombs:
        next_pos=utility.get_next_position(bomb_pos, moving_dir)
        if not utility.position_on_board(obs['board'], next_pos):
            continue
        if utility.position_is_passage(obs['board'], next_pos):
            board[next_pos]=board[bomb_pos]
            bomb_life[next_pos]=bomb_life[bomb_pos]
            bomb_blast_strength[next_pos]=bomb_blast_strength[bomb_pos]
            board[bomb_pos]=constants.Item.Passage.value
            bomb_life[bomb_pos]=0
            bomb_blast_strength[bomb_pos]=0
    return obs

def _all_directions(exclude_stop=True):
    dirs=[constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down]
    return dirs if exclude_stop else dirs + [constants.Action.Stop]

def _all_bomb_real_life(board, bomb_life, bomb_blast_st):
    def get_bomb_real_life(bomb_position, bomb_real_life):
        """One bomb's real life is the minimum life of its adjacent bomb. 
           Not that this could be chained, so please call it on each bomb mulitple times until
           converge
        """
        dirs=_all_directions(exclude_stop=True)
        min_life=bomb_real_life[bomb_position]
        for d in dirs:
            pos=bomb_position
            last_pos=bomb_position
            while True:
                pos=utility.get_next_position(pos, d)
                if _stop_condition(board, pos):
                    break
                if bomb_real_life[pos] > 0:
                    if bomb_real_life[pos]<min_life and \
                        _manhattan_distance(pos, last_pos) <= bomb_blast_st[pos]-1:
                        min_life = bomb_real_life[pos]
                        last_pos=pos
                    else:
                        break
        return min_life
    bomb_real_life_map=np.copy(bomb_life) 
    sz=len(board)
    while True:
        no_change=[]
        for i in range(sz):
            for j in range(sz):
                if utility.position_is_wall(board, (i,j)) or utility.position_is_powerup(board, (i,j)) \
                    or utility.position_is_fog(board, (i,j)):
                    continue
                if bomb_life[i,j] < 0+EPSILON:
                    continue
                real_life=get_bomb_real_life((i,j), bomb_real_life_map)
                no_change.append(bomb_real_life_map[i,j] == real_life)
                bomb_real_life_map[i,j]=real_life
        if all(no_change):
            break
    return bomb_real_life_map

def _manhattan_distance(pos1, pos2):
    #the manhattan distance here is for the specific case
    assert(pos1[0]==pos2[0] or pos1[1]==pos2[1])
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

def _stop_condition(board, pos, exclude_agent=True):
    if not utility.position_on_board(board, pos):
        return True
    if utility.position_is_fog(board, pos):
        return True
    if utility.position_is_wall(board, pos):
        return True
    if not exclude_agent:
        if utility.position_is_agent(board, pos):
            return True
    return False

def _position_covered_by_bomb(obs, pos, bomb_real_life_map):
    #return a tuple (True/False, min_bomb_life_value, max life value)
    min_bomb_pos, max_bomb_pos=None, None
    min_bomb_value, max_bomb_value=INT_MAX, -INT_MAX
    if obs['bomb_life'][pos]>0:
        min_bomb_value, max_bomb_value=bomb_real_life_map[pos],bomb_real_life_map[pos]
        min_bomb_pos, max_bomb_pos=pos, pos
    dirs=_all_directions(exclude_stop=True)
    board=obs['board']
    for d in dirs:
        next_pos=pos
        while True:
            next_pos=utility.get_next_position(next_pos, d)
            if _stop_condition(board, next_pos, exclude_agent=True):
                #here should assume agents are dynamic
                break
            if obs['bomb_life'][next_pos]>0 and obs['bomb_blast_strength'][next_pos] - 1 >= _manhattan_distance(pos, next_pos):
                if bomb_real_life_map[next_pos] < min_bomb_value:
                    min_bomb_value=bomb_real_life_map[next_pos]
                    min_bomb_pos=next_pos
                if bomb_real_life_map[next_pos] > max_bomb_value:
                    max_bomb_value=bomb_real_life_map[next_pos]
                    max_bomb_pos = next_pos
                break
    if min_bomb_pos is not None:
        return True, min_bomb_value, max_bomb_value
    return False, INT_MAX, -INT_MAX
  
def _compute_min_evade_step(obs, parent_pos_list, pos, bomb_real_life):
    flag_cover, min_cover_value, max_cover_value=_position_covered_by_bomb(obs, pos, bomb_real_life)
    if not flag_cover:    
        return 0
    elif len(parent_pos_list) >= max_cover_value:
        if len(parent_pos_list)   > max_cover_value + FLAME_LIFE :
            return 0
        else:
            return INT_MAX
    elif len(parent_pos_list)  >= min_cover_value:
        if len(parent_pos_list)  > min_cover_value + FLAME_LIFE:
            return 0
        else:
            return INT_MAX
    else:
        board=obs['board']
        dirs=_all_directions(exclude_stop=True)
        min_step=INT_MAX
        for d in dirs:
            next_pos=utility.get_next_position(pos, d)
            if not utility.position_on_board(board, next_pos):
                continue
            if not (utility.position_is_passage(board, next_pos) or \
            utility.position_is_powerup(board, next_pos)):
                continue
            if next_pos in parent_pos_list:
                continue
            x=_compute_min_evade_step(obs, parent_pos_list+[next_pos], next_pos, bomb_real_life)
            min_step=min(min_step, x+1)
        return min_step

def _check_if_flame_will_gone(obs, prev_two_obs, flame_pos):
    assert(prev_two_obs[0] is not None)
    assert(prev_two_obs[1] is not None)
    #check the flame group in current obs, see if 
    #the whole group was there prev two obs
    #otherwise, although this flame appears in prev two obs, 
    #it could be a old overlap new, thus will not gone next step
    if not (utility.position_is_flames(prev_two_obs[0]['board'], flame_pos) \
        and utility.position_is_flames(prev_two_obs[1]['board'], flame_pos)):
        return False
    board=obs['board']
    Q=deque(maxlen=121)
    Q.append(flame_pos)
    visited=[flame_pos]
    dirs=_all_directions(exclude_stop=True)
    while len(Q)>0:
        pos=Q.popleft()
        if not (utility.position_is_flames(prev_two_obs[0]['board'], pos) \
            and utility.position_is_flames(prev_two_obs[1]['board'], pos)):
            return False
        for d in dirs:
            next_pos=utility.get_next_position(pos, d)
            if utility.position_on_board(board, next_pos) and utility.position_is_agent(board, next_pos):
                if next_pos not in visited:
                    Q.append(next_pos)
                    visited.append(next_pos)
    return True

def _compute_safe_actions(obs, exclude_kicking=False, prev_two_obs=(None,None)):
    dirs=_all_directions(exclude_stop=True)
    ret=set()
    my_position, board, blast_st, bomb_life, can_kick=obs['position'], obs['board'], obs['bomb_blast_strength'], obs['bomb_life'], obs['can_kick']
    kick_dir=None
    bomb_real_life_map=_all_bomb_real_life(board, bomb_life, blast_st)
    flag_cover_passages=[]
    for direction in dirs:
        position = utility.get_next_position(my_position, direction) 
        if not utility.position_on_board(board, position):
            continue
        if (not exclude_kicking) and utility.position_in_items(board, position, [constants.Item.Bomb]) and can_kick:
            #filter kick if kick is unsafe
            if _kick_test(board, blast_st, bomb_real_life_map, my_position, direction): 
                ret.add(direction.value)
                kick_dir=direction.value
        gone_flame_pos=None
        if (prev_two_obs[0]!=None and prev_two_obs[1]!=None) and _check_if_flame_will_gone(obs, prev_two_obs, position):
            #three consecutive flames means next step this position must be good
            #make this a candidate
            obs['board'][position]=constants.Item.Passage.value
            gone_flame_pos=position

        if utility.position_is_passage(board, position) or utility.position_is_powerup(board, position):
            my_id=obs['board'][my_position]
            obs['board'][my_position]= constants.Item.Bomb.value if bomb_life[my_position]>0  else constants.Item.Passage.value
            flag_cover, min_cover_value , max_cover_value=_position_covered_by_bomb(obs, position, bomb_real_life_map)
            flag_cover_passages.append(flag_cover)
            if not flag_cover:
                ret.add(direction.value)
            else:
                min_escape_step=_compute_min_evade_step(obs, [position], position, bomb_real_life_map)
                assert(min_escape_step>0)
                if min_escape_step  < min_cover_value:
                    ret.add(direction.value)
            obs['board'][my_position]=my_id
        if gone_flame_pos is not None:
            obs['board'][gone_flame_pos]=constants.Item.Flames.value

    # Test Stop action only when agent is covered by bomb, 
    # otherwise Stop is always an viable option 
    my_id=obs['board'][my_position]
    obs['board'][my_position]= constants.Item.Bomb.value if bomb_life[my_position]>0  else constants.Item.Passage.value    
    #REMEMBER: before compute min evade step, modify board accordingly first..
    flag_cover, min_cover_value, max_cover_value=_position_covered_by_bomb(obs, my_position, bomb_real_life_map)
    if flag_cover:
        min_escape_step=_compute_min_evade_step(obs, [None, my_position], my_position, bomb_real_life_map)
        if min_escape_step < min_cover_value:
            ret.add(constants.Action.Stop.value)
    else:
        ret.add(constants.Action.Stop.value)
    obs['board'][my_position]=my_id
    #REMEBER: change the board back

    #Now test bomb action
    if not (obs['ammo'] <=0 or obs['bomb_life'][obs['position']]>0):
        #place bomb might be possible
        assert(BOMBING_TEST in ['simple', 'simple_adjacent', 'lookahead'])
        if BOMBING_TEST == 'simple':
            if not flag_cover:
                ret.add(constants.Action.Bomb.value)
        elif BOMBING_TEST == 'simple_adjacent':
            if (not flag_cover) and (not any(flag_cover_passages)): 
                ret.add(constants.Action.Bomb.value)
        else: #lookahead
            if (constants.Action.Stop.value in ret) and len(ret)>1 and (kick_dir is None):
                obs2=copy.deepcopy(obs)
                my_pos=obs2['position']
                obs2['board'][my_pos]=constants.Item.Bomb.value
                obs2['bomb_life'][my_pos]=min_cover_value if flag_cover else 10
                obs2['bomb_blast_strength'][my_pos]=obs2['blast_strength']
                bomb_life2, bomb_blast_st2, board2=obs2['bomb_life'], obs2['bomb_blast_strength'],obs2['board']
                bomb_real_life_map=_all_bomb_real_life(board2, bomb_life2, bomb_blast_st2)
                min_evade_step=_compute_min_evade_step(obs2, [None, my_position], my_pos, bomb_real_life_map)
                current_cover_value=obs2['bomb_life'][my_pos]
                if min_evade_step  < current_cover_value:
                    ret.add(constants.Action.Bomb.value)
    return ret

def get_filtered_actions(obs, prev_two_obs=None):
    if obs['board'][obs['position']] not in obs['alive']:
        return [constants.Action.Stop.value]
    obs_cpy=copy.deepcopy(obs)
    if prev_two_obs[-1] is not None:
        obs=move_moving_bombs_to_next_position(prev_two_obs[-1], obs)
    ret=_compute_safe_actions(obs,exclude_kicking=NO_KICKING, prev_two_obs=prev_two_obs)
    obs=obs_cpy
    if len(ret)!=0:
        return list(ret)
    else:
        return [constants.Action.Stop.value]

def _kick_test(board, blast_st, bomb_life, my_position, direction):
    def moving_bomb_check(moving_bomb_pos, p_dir, time_elapsed):
        pos2=utility.get_next_position(moving_bomb_pos, p_dir)
        dist=0
        for i in range(10):
            dist +=1
            if not utility.position_on_board(board, pos2):
                break
            if not (utility.position_is_powerup(board, pos2) or utility.position_is_passage(board, pos2)):
                break
            life_now=bomb_life[pos2] - time_elapsed
            if bomb_life[pos2]>0 and life_now>=-2 and life_now <= 0 and dist<blast_st[pos2]:
                return False
            pos2=utility.get_next_position(pos2, p_dir)
        return True
    next_position=utility.get_next_position(my_position, direction)
    assert(utility.position_in_items(board, next_position, [constants.Item.Bomb]))
    life_value=int(bomb_life[next_position])
    strength=int(blast_st[next_position])
    dist=0
    pos=utility.get_next_position(next_position, direction)
    perpendicular_dirs=[constants.Action.Left, constants.Action.Right] 
    if direction == constants.Action.Left or direction == constants.Action.Right:
        perpendicular_dirs=[constants.Action.Down, constants.Action.Up]
    for i in range(life_value): 
        if utility.position_on_board(board, pos) and utility.position_is_passage(board, pos):
            #do a check if this position happens to be in flame when the moving bomb arrives!
            if not (moving_bomb_check(pos, perpendicular_dirs[0], i) and moving_bomb_check(pos, perpendicular_dirs[1], i)):
                break
            dist +=1 
        else:
            break
        pos=utility.get_next_position(pos, direction)
        #can kick and kick direction is valid
    return dist > strength

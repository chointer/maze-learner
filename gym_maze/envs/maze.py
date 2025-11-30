from collections import deque
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from .rewards import RewardManager

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, height_range=[5, 10], width_range=[5, 10], manual_mode=False):
        self.height_max = height_range[1]
        self.width_max = width_range[1]
        self.height_range = height_range
        self.width_range = width_range
        self.window_size = 720          # Pygame window size
        self.manual_mode = manual_mode

        # minimum value check
        if self.height_range[0] <= 1:
            raise ValueError(f"Invalid minimum height: {self.height_range[0]}. The minimum height must be greater than 1.")
        if self.width_range[0] <= 1:
            raise ValueError(f"Invalid minimum width: {self.width_range[0]}. The minimum weight must be greater than 1.")

        # Create a placeholder? for Observation Space - locations of the agent and the target
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.height_max, self.width_max, 4 + 2),
            dtype=np.float32
        )
        # self.observation_space = spaces.MultiBinary(
        #     [self.height_range[1], self.width_range[1], 4 + 2]      # 상하좌우 + 플레이어위치, 목표위치
        # )
        """ stable_baseline3에서 Dict 형태를 지원하지 않고, 배열이 아니면 인공신경망에 입력하기도 불편해서 MultiBinary로 변경.
        self.observation_space = spaces.Dict(
            {
                "maze": spaces.MultiBinary(
                    [self.height_range[1], self.width_range[1], 4]
                    ),
                "agent": spaces.Box(
                    np.array([0, 0]), 
                    np.array([self.height_range[1] - 1, self.width_range[1] - 1]), 
                    shape=(2,),
                    dtype=int
                    ),
                "target": spaces.Box(
                    np.array([0, 0]), 
                    np.array([self.height_range[1] - 1, self.width_range[1] - 1]), 
                    shape=(2,),
                    dtype=int
                    )
            }
        )
        """
        # Create a placehoder for Action Space - "Up", "Down", "Left", "Right", "NoAction" (= 4)
        self.action_space = spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array((-1, 0), dtype=int),        # Up
            1: np.array((1, 0), dtype=int),         # Down
            2: np.array((0, -1), dtype=int),        # Left
            3: np.array((0, 1), dtype=int),         # Right
        }
        
        # Reward
        #self.distance_map = None
        #self.manhattan_map = None
        self.reward = 0
        self.reward_manager = RewardManager()

        # Rendering settings
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Create placeholders for render_mode="human"
        self.window = None      # a reference(참조) to the window that we draw to
        self.clock = None       # clock for correct framerate
        self.font = None


    def _get_obs(self):
        locs = np.zeros((self.height_max, self.width_max, 2), dtype=np.float32)
        locs[self._agent_location[0], self._agent_location[1], 0] = 1.
        locs[self._target_location[0], self._target_location[1], 1] = 1.
        return np.concatenate([self.maze_fullsize, locs], axis=-1).astype(np.float32)

        #return {
        #    "maze": self.maze_fullsize, 
        #    "agent": self._agent_location, 
        #    "target": self._target_location
        #}


    def _get_info(self):
        return {
            "move_count": self.move_count, 
            "agent_location": self._agent_location,
            "target_location": self._target_location,
            "maze_h": self.maze_height,
            "maze_w": self.maze_width,
        }  #, "distance_map": self.distance_map, "manhattan_map": self.manhattan_map}


    def set_generation_size(self, h_range, w_range):
        # value check
        if h_range[0] <= 1:
            raise ValueError(f"Invalid minimum height: {h_range[0]}. The minimum height must be greater than 1.")
        if h_range[1] > self.height_max:
            raise ValueError(f"Invalid maximum height: {h_range[1]}. The maximum height must be <= {self.height_max}.")
        if w_range[0] <= 1:
            raise ValueError(f"Invalid minimum width: {w_range[0]}. The minimum weight must be greater than 1.")
        if w_range[1] > self.width_max:
            raise ValueError(f"Invalid minimum width: {w_range[1]}. The maximum weight must be <= {self.width_max}.")
        
        self.height_range = h_range
        self.width_range = w_range


    def set_manual(self, value):
        if not isinstance(value, bool):
            raise ValueError("Input must be a boolean value")
        self.manual_mode = value


    def reset(self, seed=None, nr_ratio=0.75, options=None):
        super().reset(seed=seed)

        # initialize maze size       
        self.maze_height = self.np_random.integers(self.height_range[0], self.height_range[1])
        self.maze_width = self.np_random.integers(self.width_range[0], self.width_range[1])

        # make maze
        self.maze = self.generate_maze(self.maze_height, self.maze_width, nr_ratio=nr_ratio)
        # self.maze_fullsize: observation_space["maze"]의 형태에 맞춘 maze array
        maze_ones = np.ones((self.height_max, self.width_max, 4))
        maze_ones[:self.maze_height, :self.maze_width] = self.maze
        self.maze_fullsize = maze_ones

        # initialize object locations
        self._agent_location = self.np_random.integers([self.maze_height, self.maze_width], size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers([self.maze_height, self.maze_width], size=2, dtype=int)

        self.move_count = 0

        # RETURNS
        observation = self._get_obs()
        info = self._get_info()
        
        # REWARD INPUTS
        self.distance_map = self.compute_distance_map(observation)
        self.manhattan_map = self.compute_manhattan_map(observation)
        self.reward_manager.reward_inputs['distance_map'] = self.distance_map
        self.reward_manager.reward_inputs['manhattan_map'] = self.manhattan_map
        self.reward_manager.reward_inputs['agent_location'] = self._agent_location

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    def step(self, action):
        # action {0, 1, 2, 3, 4}: {Up Down Left Right NoAction}
        if action != 4 and not self.maze[self._agent_location[0], self._agent_location[1], action]:
            direction = self._action_to_direction[action]

            self._agent_location = np.clip(
                self._agent_location + direction, (0, 0), (self.maze_height - 1, self.maze_width - 1)
            )

        # End condition
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        self.reward_manager.reward_inputs['terminated'] = terminated
        self.reward_manager.reward_inputs['action'] = action
        self.reward_manager.reward_inputs['manual_mode'] = self.manual_mode
        self.reward_manager.reward_inputs['agent_location_before'] = self.reward_manager.reward_inputs['agent_location'].copy()     # 혹시 모를 데이터 공유 오류 방지
        self.reward_manager.reward_inputs['agent_location'] = self._agent_location
        reward, reward_dict = self.reward_manager()
        # reward = 1 if terminated else 0
        
        if self.manual_mode is False or action != 4:
            self.move_count += 1
            self.reward = reward
        #     reward -= 0.01

        observation = self._get_obs()
        info = self._get_info()
        info["manual_mode"] = self.manual_mode
        info["rewards"] = reward_dict

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info


    def generate_maze(self, height, width, nr_ratio=0.75):
        wall_c = np.ones((height, width, 4), dtype=bool)
        cell_visited = np.zeros((height, width), dtype=bool)
        cell_stack = []

        # [step 1] Select initial cell and stack.
        #cell_current = self.np_random.integers([self.maze_height, self.maze_width], size=2, dtype=int) # modified: np.random.randint -> np_random.integers
        cell_current = self.np_random.integers([height, width], size=2, dtype=int) # modified: np.random.randint -> np_random.integers
        cell_stack.append(cell_current)
        cell_visited[cell_current[0], cell_current[1]] = True
        
        # [step 2] Remove wall
        while cell_stack:
            # [step 2.1] Select cell_current; Newest vs. Random
            if self.np_random.random() <= nr_ratio:    # modified: np.random.rand -> self.np_random.random
                cell_current_idx = -1       # Newest; Recursive Backtracking
            else:
                cell_current_idx = self.np_random.integers(len(cell_stack))       # Random; Prim's Algorithm, modified: np.random.randint -> self.np_random.integers
            cell_current = cell_stack[cell_current_idx]

            # [step 2.2] Check adjacent cells & remove wall
            #for direction in np.random.permutation(["Up", "Down", "Left", "Right"]):
            for direction_idx in self.np_random.permutation(4):        # modified: np.random.permutation(["Up", "Down", "Left", "Right"]) -> self.np_random.permutation(4)
                direction = self._action_to_direction[direction_idx]   # direction_idx {0, 1, 2, 3} = direction {up, down, left, right}
                cell_next = cell_current + direction    #self.dir2move[direction]
                # [2.2.a] 방문하지 않은 cell이면, stack cell_next, visit check, remove wall
                if cell_next[0] >= 0 and cell_next[0] < height and cell_next[1] >= 0 and cell_next[1] < width and not cell_visited[cell_next[0], cell_next[1]]:
                    cell_stack.append(cell_next)
                    cell_visited[cell_next[0], cell_next[1]] = True
                    wall_c[cell_current[0], cell_current[1], direction_idx] = False
                    direction_idx_opposite = 2*(direction_idx//2) + (direction_idx + 1)%2
                    wall_c[cell_next[0], cell_next[1], direction_idx_opposite] = False
                    break
                # [2.2.b] (no code) 이미 방문된 셀이면, 다음 방향 순서로 반복

            # [2.2.c] 더 이상 방문할 셀이 없으면, stack에서 현재 cell 제거. break 없이 for 반복을 완료한 상태이므로 else로 진입하게 된다.
            else:
                del cell_stack[cell_current_idx]

        return wall_c


    def maze_c2w(self, wall_c):     # from (default) cell type to wall type
        height, width, _ = wall_c.shape
        wall_h = np.ones((height + 1, width), dtype=bool)
        wall_v = np.ones((height, width + 1), dtype=bool)
        
        for i, line in enumerate(wall_c):
            for j, walls in enumerate(line):
                if not walls[0]:            # 0: Up
                    wall_h[i][j] = False
                if not walls[2]:            # 2: Left
                    wall_v[i][j] = False
                # Edge Conditions
                if i == height - 1 and not walls[1]:        # 1: Down
                    wall_h[i + 1][j] = False
                if j == width - 1 and not walls[3]:         # 3: Right
                    wall_v[i][j + 1] = False

        return wall_h, wall_v
    

    def maze_w2c(self, wall_h, wall_v):
        height, width = wall_h.shape
        height -= 1
        wall_c = np.ones((height, width, 4), dtype=bool)

        # Carve Horizontal
        for i, line in enumerate(wall_h[1:-1]):
            for j, wall in enumerate(line):
                if not wall:
                    wall_c[i][j][1] = False             # 1: Down
                    wall_c[i + 1][j][0] = False         # 0: Up
                    # 0, -1번째 줄은 반복문에서 없으므로, 항상 (i + 1)th cell은 존재할 것.
        
        # Carve Vertical
        for i, line in enumerate(wall_v):
            for j, wall in enumerate(line[1:-1]):
                if not wall:
                    wall_c[i][j][3] = False             # 3: Right
                    wall_c[i][j + 1][2] = False         # 2: Left

        # Carve Edge Walls; 꼭 할 필요는 없지만, 혹시 확장하면서 버그가 생길지 모르니까 추가
        for i, wall in enumerate(wall_h[0]):
            if not wall: 
                wall_c[0][i][0] = False                 # 0: Up
        for i, wall in enumerate(wall_h[-1]):
            if not wall:
                wall_c[-1][i][1] = False                # 1: Down
        for i, wall in enumerate(wall_v[:, 0]):
            if not wall: 
                wall_c[i][0][2] = False                 # 2: Left
        for i, wall in enumerate(wall_v[:, -1]):
            if not wall: 
                wall_c[i][-1][3] = False                # 3: Right

        return width, height, wall_c


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        

    def _render_frame(self):
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("arial", 20)

        # Initialization in "human" mode
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Generate a canvas
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / max(self.height_max, self.width_max, 22))

        startx = (self.window_size - self.maze_width * pix_square_size)/2
        starty = (self.window_size - self.maze_height * pix_square_size)/2

        # Draw maze
        wall_h, wall_v = self.maze_c2w(self.maze)
        for y, row in enumerate(wall_h):
            for x, wall in enumerate(row):
                if wall:
                    pygame.draw.line(
                        canvas,
                        0,
                        (startx + pix_square_size * x, starty + pix_square_size * y),
                        (startx + pix_square_size * (x + 1), starty + pix_square_size * y),
                        width=3,
                    )

        for y, row in enumerate(wall_v):
            for x, wall in enumerate(row):
                if wall:
                    pygame.draw.line(
                        canvas,
                        0,
                        (startx + pix_square_size * x, starty + pix_square_size * y),
                        (startx + pix_square_size * x, starty + pix_square_size * (y + 1)),
                        width=3,
                    )

        # Draw objects
        # target
        pygame.draw.rect(
            canvas,
            (255, 80, 80),
            pygame.Rect(
                startx + self._target_location[1] * pix_square_size + pix_square_size*0.2,
                starty + self._target_location[0] * pix_square_size + pix_square_size*0.2,
                pix_square_size * 0.6, 
                pix_square_size * 0.6
            )
        )
        # agent
        pygame.draw.circle(
            canvas,
            (0, 80, 255),
            (startx + (self._agent_location[1] + 0.5) * pix_square_size, starty + (self._agent_location[0] + 0.5) * pix_square_size),
            pix_square_size / 4,
        )

        # Draw text
        board = self.font.render(f"Mvs: {self.move_count}", True, (0, 0, 0))
        canvas.blit(board, [startx, starty - 1.8 * pix_square_size])
        board = self.font.render(f"Rwd: {round(self.reward, 3)}", True, (0, 0, 0))
        canvas.blit(board, [startx, starty - pix_square_size])

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            #pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def get_draw_infos(self):
        pix_square_size = (self.window_size / max(self.height_max, self.width_max, 22))
        return {
            'pix_square_size': pix_square_size,
            'startx': float(self.window_size - self.maze_width * pix_square_size)/2,
            'starty': float(self.window_size - self.maze_height * pix_square_size)/2
        }

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_keys_to_action(self):
        return {"w": 0, "s": 1, "a": 2, "d": 3}
    
    def compute_distance_map(self, obs=None):
        # BFS
        if obs is None:
            obs = self._get_obs()
        height, width, _ = obs.shape
        wall_c = obs[:, :, :4].astype(int)
        dists = np.ones((height, width), dtype=int) * -1

        target_loc_y, target_loc_x = np.where(obs[:, :, -1])    
        q = deque()
        q.append((target_loc_y[0], target_loc_x[0], 0))     # y, x, distance

        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # obs[:, :, idx]의 0~3 idx는 순서대로 Up Down Left Right를 의미한다.

        while q:
            y, x, d = q.popleft()
            if dists[y, x] != -1:       # 첫 방문이 아니면 continue
                continue
            
            # 알고리즘 오류 체크
            if dists[y, x] > d:
                print(f'y={y}, x={x}, d={d}')
                print(dists)
                assert False

            # 거리 기입
            dists[y, x] = d

            # 다음 방문지 선택
            for idx_direction in range(4):  # up down left right
                y_next = y + MOVES[idx_direction][0]
                x_next = x + MOVES[idx_direction][1]
                if wall_c[y, x, idx_direction] == 0 and dists[y_next, x_next] == -1:       # No wall(movable) & Not Visited
                    q.append((y_next, x_next, d + 1))

        return dists

    def compute_manhattan_map(self, obs=None):
        h_max, w_max, _ = obs.shape
        ws = np.arange(w_max)
        hs = np.arange(h_max)
        ww, hh = np.meshgrid(ws, hs)
        target_loc_y, target_loc_x = np.where(obs[:, :, -1])
        manhattan = np.abs(ww - target_loc_x) + np.abs(hh - target_loc_y)
        return manhattan

#env = MazeEnv(render_mode="rgb_array", height_range=[15, 20], width_range=[15, 20])

#from gymnasium.utils.play import play
#play(env, callback=f, noop=-1)
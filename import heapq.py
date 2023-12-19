import heapq
import time
from operator import indexOf
import random

# Cài đặt chung đầu bài cho các thuật toán về bàn cờ và cây tìm kiếm
class NQueens:
    def __init__(self, n, goal_state=None):
        self.init_state = tuple([-1] * n)
        self.goal_state = goal_state
        self.n = n

    def actions(self, state):
        if state[-1] != -1:  # nếu mọi cột đều được fill đầy
            return []  # không có hành động được trả về

        valid_actions = list(range(self.n))
        col = state.index(-1)  
        for row in range(self.n):
            for c, r in enumerate(state[:col]):
                if self.conflict(row, col, r, c) and row in valid_actions:
                    valid_actions.remove(row)
        return valid_actions

    def result(self, state, action):
        col = state.index(-1)  # làm trống cột trái nhất, cột ngoài
        new = list(state[:])
        new[col] = action  # quân hậu nằm trên cột này
        return tuple(new)

    def final_solution(self, state):
        try:
            if state[-1] == -1:  # nếu là cột rỗng
                return False  # vậy chỗ này không phải chỗ cần tìm
        except IndexError:  
            return True
        for c1, r1 in enumerate(state):
            for c2, r2 in enumerate(state):
                if (r1, c1) != (r2, c2) and self.conflict(r1, c1, r2, c2):
                    return False
        return True

    def conflict(self, row1, col1, row2, col2):
        return row1 == row2 or col1 == col2 or abs(row1 - row2) == abs(col1 - col2)

    def g(self, cost, from_state, action, to_state):
        return cost + 1

    def h(self, state):
        conflicts = 0
        for col1, row1 in enumerate(state):
            for col2, row2 in enumerate(state[col1 + 1:], start=col1 + 1):
                if self.conflict(row1, col1, row2, col2):  
                    conflicts += 2
        return conflicts


class Node: #tạo Node trên cây tìm kiếm
    def __init__(self, state, parent=None, action=None,path_cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def expand(self, Queen):  # danh sách mấy cái node con ạ
        return [self.child_node(Queen, action)
                for action in Queen.actions(self.state)]

    def child_node(self, Queen, action):  
        pass

    def solution(self):  # trả về node root từ cái node hiện tại
        if self.state is None:
            return None
        return [node.action for node in self.path()[1:]]

    def path(self):  # trả về nguyên cái danh sách(liên kết) cho node root từ cái node hiện tại
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


  #Phần này dành cho UCS ALGORITHM
def UCS(Queen):
    node = NodeU(Queen.init_state)
    return Searching(node, Queen)

class NodeU(Node):  # NodeU là dành cho UCS Algorithm
    def __init__(self, state, parent=None, action=None, path_cost=0):
        super().__init__(state, parent, action, path_cost)

    def child_node(self, Queen, action):
        next_state = Queen.result(self.state, action)
        next_node = NodeU(next_state, self, action, Queen.g(self.path_cost, self.state, action, next_state))
        return next_node

    def __lt__(self, other):
        return self.path_cost < other.path_cost


  #Phần này dành cho A* Algorithm
def AS(Queen):
    node = NodeA(Queen.init_state, heuristic=Queen.h(Queen.init_state))
    return Searching(node, Queen)


class NodeA(Node):  # NodeA là node dành cho A* Algorithm
    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic=0):
        super().__init__(state, parent, action, path_cost, heuristic)

    def child_node(self, Queen, action):
        next_state = Queen.result(self.state, action)
        next_node = NodeA(next_state, self, action, Queen.g(self.path_cost, self.state, action, next_state), Queen.h(next_state))
        return next_node

    def __lt__(self, other):
        return (self.path_cost + self.heuristic) < (other.path_cost + other.heuristic)

 # Phần này dành cho thuật toán genetic
def genetic_queen(population, MaximumFit):
      mutation_probability = 0.1
      new_population = []
      sorted_population = []
      probabilities = []
      for n in population:
          f = BoardFit(n, MaximumFit)
          probabilities.append(f / MaximumFit)
          sorted_population.append([f, n])

      sorted_population.sort(reverse=True)

      new_population.append(sorted_population[0][1])  # di truyền tốt nhất
      new_population.append(sorted_population[-1][1])  # di truyền xấu nhất

      for i in range(len(population) - 2):

          Queen_posi_1 = random_pick(population, probabilities)
          Queen_posi_2 = random_pick(population, probabilities)

          # tạo ra 2 vị trí hậu mới
          child = crossover(Queen_posi_1, Queen_posi_2)

          if random.random() < mutation_probability:
              child = mutate(child)

          new_population.append(child)
          if BoardFit(child, MaximumFit) == MaximumFit:
              break
      return new_population

# Phần Search
def Searching(node, Queen):
    frontier = [node]
    heapq.heapify(frontier)
    expanded = [Queen.init_state]
    while frontier:
        current = heapq.heappop(frontier)
        
        if Queen.final_solution(current.state):  # tìm được kết quả
            return current
        if current in expanded:
                   continue
        children = current.expand(Queen)  
        expanded.append(current)
        for i in children:
            if i not in expanded:
                heapq.heappush(frontier, i)
    return Node(0, None, None)  # nếu cái frontier mà rỗng thì hong có kết quả


print("enter N queen: ")
nq=int(input())
print("choose algorithm - type 1 for UCS or type 2 for A* or type 3 for genetic : ")
y=int(input())
if y==2:
  start_time = time.time()
  board = NQueens(nq)  
  a_star = AS(board)
  print('A*: ' + str(a_star.solution()))
  end_time= time.time()
  runtime= end_time-start_time
  print(runtime)

if y==1:
  start_time = time.time()
  board = NQueens(nq);
  uniform_cost = UCS(board)
  print('Uniform-Cost: ' + str(uniform_cost.solution()))
  end_time= time.time()
  runtime1= end_time-start_time
  print(runtime1)

if y==3:
  start_time = time.time()
  def random_Queen_posi(size):
      return [random.randint(0, size - 1) for _ in range(size)]


  def BoardFit(Queen_posi, MaximumFit):
      horizontal_collisions = (
          sum([Queen_posi.count(queen) - 1 for queen in Queen_posi]) / 2
      )
      diagonal_collisions = 0

      n = len(Queen_posi)
      left_diagonal = [0] * (2 * n - 1)
      right_diagonal = [0] * (2 * n - 1)
      for i in range(n):
          left_diagonal[i + Queen_posi[i] - 1] += 1
          right_diagonal[len(Queen_posi) - i + Queen_posi[i] - 2] += 1

      diagonal_collisions = 0
      for i in range(2 * n - 1):
          counter = 0
          if left_diagonal[i] > 1:
              counter += left_diagonal[i] - 1
          if right_diagonal[i] > 1:
              counter += right_diagonal[i] - 1
          diagonal_collisions += counter

   
      return int(MaximumFit - (horizontal_collisions + diagonal_collisions))

  def crossover(x, y):
      n = len(x)
      child = [0] * n
      for i in range(n):
          c = random.randint(0, 1)
          if c < 0.5:
              child[i] = x[i]
          else:
              child[i] = y[i]
      return child


  # ngẫu nhiên thay đổi giá trị của một cái index ngẫu nhiên về vị trí hậu
  def mutate(x):
      n = len(x)
      c = random.randint(0, n - 1)
      m = random.randint(0, n - 1)
      x[c] = m
      return x


  # Lựa chọn (cũng ngẫu nhiên)
  def random_pick(population, probabilities):
      populationWithProbabilty = zip(population, probabilities)
      total = sum(w for c, w in populationWithProbabilty)
      r = random.uniform(0, total)
      upto = 0
      for c, w in zip(population, probabilities):
          if upto + w >= r:
              return c
          upto += w
      assert False, "Shouldn't get here"

  # in ra màn hình vị trí hậu đã tìm được
  def print_Queen_posi(chrom, MaximumFit):
      print(
          "Genetic Algorithm = {}".format(str(chrom))
      )
  # In nguyên cái bàn ờ có hậu ra
  def print_board(chrom):
      board = []

      for x in range(nq):
          board.append(["x"] * nq)

      for i in range(nq):
          board[chrom[i]][i] = "Q"

      def print_board(board):
          for row in board:
              print(" ".join(row))

      print()
      print_board(board)


  if __name__ == "__main__":
          POPULATION_SIZE = 500    
          
          MaximumFit = (nq * (nq - 1)) / 2  
          population = [random_Queen_posi(nq) for _ in range(POPULATION_SIZE)]

          Gen = 1
          while (
              not MaximumFit in [BoardFit(chrom, MaximumFit) for chrom in population]
              and Gen < 200
          ):

              population = genetic_queen(population, MaximumFit)
              if Gen % 10 == 0:
                  format(Gen)
                  format(
                          max([BoardFit(n, MaximumFit) for n in population])
                      )
                  
              Gen += 1

          BoardFitOfQueen_posis = [BoardFit(chrom, MaximumFit) for chrom in population]

          bestQueen_posis = population[
              indexOf(BoardFitOfQueen_posis, max(BoardFitOfQueen_posis))
          ]

          if MaximumFit in BoardFitOfQueen_posis:
              format(Gen - 1)

              print_Queen_posi(bestQueen_posis, MaximumFit)

              print_board(bestQueen_posis)

          else:
              format(
                      Gen - 1
                  )
              
              print_board(bestQueen_posis)
  end_time= time.time()
  runtime2= end_time-start_time
  print(runtime2)
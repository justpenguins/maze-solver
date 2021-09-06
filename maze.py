import numpy
import cv2
import matplotlib.pyplot as plt

# Define a vertex object with the following fields
class Vertex():

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.distance = int('inf')
        self.parent_x = None
        self.parent_y = None
        self.processed = False
        self.place_in_queue = None


# Gets neighbours after checking that the neighbour is within photo/maze
def neighbours(matrix, rows, cols):
    neighbours = []
    shape = matrix.shape

    if cols > 0 and not matrix[rows][cols - 1].processed:
        neighbours.append(matrix[rows][cols - 1])
    if cols < shape[1] - 1 and not matrix[rows][cols + 1].processed:
        neighbours.append(matrix[rows][cols - 1])
    if rows > 0 and not matrix[rows - 1][cols].processed:
        neighbours.append(matrix[rows - 1][cols])
    if rows < shape[0] - 1 and not matrix[rows + 1][cols].processed:
        neighbours.append(matrix[rows][cols + 1])
    return neighbours

# Get Euclidean^2 distance (length of line between 2 points)
# Finds distance between neighbours, using neighbours function
def get_distance(img, x, y):
    return 0.1 + (float(img[y][0])-float(img[x][0]))**2 + (float(img[y][1]) - float(img[x][1]))**2 + (float(img[y][2]) - float(img[x][2]))**2


# Helper function to draw the lines of the path
def draw(img, route):
    x = route[0]
    y = route[0]

    for vertex in route[1:]:
        x, y, x1, y1 = vertex
        cv2.line(img, (x,y), (x1,y1), (255,0,0), 2)


# Helper for processing the indexes of nodes (aka the path of the maze)
def index_swap(queue, index):
    new_index = ((index - 1) / 2)
    if index <= 0:
        return queue
    if queue[index].distance < queue[new_index].distance:
        queue[new_index] = queue[index]
        queue[index] = queue[new_index]
        queue[new_index].place_in_queue = new_index
        queue[index].place_in_queue = index
        queue = index_swap(queue, new_index)
    return queue


# Helper for processing path of the children
def child_swap(index, queue):
    left_child = (index * 2) + 1 # Index in list of left child
    right_child = index + 1      # Index in list of right child
    length = len(queue)

    if left_child >= length:
        return queue
    elif left_child < length and right_child >= length:
        if queue[index].distance > queue[left_child].distance:
            queue[index] = queue[left_child]
            queue[left_child] = queue[index]
            queue[left_child].place_in_queue = left_child
            queue[index].place_in_queue = index
            queue = child_swap(queue, left_child)
    else:
        smol_index = left_child
        if queue[left_child].distance > queue[right_child].distance:
            smol_index = right_child
        elif queue[smol_index].distance > queue[right_child].distance:
            queue[index] = queue[smol_index]
            queue[smol_index] = queue[index]
            queue[smol_index].place_in_queue = smol_index
            queue[index].place_in_queue = index
            queue = child_swap(queue, smol_index)
    return queue
        


# Main driver function
#
# Takes start and end coordinate 
# Checks to see that the coords are within bounds
# Calls child/index swap to solve the maze
#
def solver(img, start, end):
    start_x = start[0]
    start_y = start[1]
    end_x = end[0]
    end_y = end[1]
    priority_queue = []

    img_rows = img.shape[0]
    img_cols = img.shape[1]

    matrix = numpy.full((img_rows, img_cols), None)
    
    for cols in range(img_cols):
        for rows in range(img_rows):
            matrix[rows][cols] = Vertex(cols, rows)
            matrix[rows][cols].place_in_queue = len(priority_queue)
            priority_queue.append(matrix[rows][cols])

    matrix[start_x][start_y].distance = 0
    mat_index = matrix[start_x][start_y].place_in_queue
    priority_queue = child_swap(priority_queue, mat_index)

    while len(priority_queue) > 0: # While things are in the pq
        node.processed = True      # Set the nodes to be processed
        node = priority_queue[0]   # Set to front of queue
        node.place_in_queue = 0    # Set "pointer" to front
        priority_queue[0] = priority_queue[-1] # Previous front is not the one before the front
        priority_queue.pop()       # Remove the one before the front
        neighbours = neighbours(matrix, node.y, node.x)

        for vertex in neighbours:
            distance = get_distance(img, (node.y, node.x), (vertex.y, vertex.x))
            if node.distance + distance < vertex.distance:
                vertex.distance = node.distance + distance
                vertex.parent_y = node.y
                vertex.parent_x = node.x

                v_index = vertex.place_in_queue
                priority_queue = index_swap(priority_queue, v_index)
                priority_queue = child_swap(priority_queue, v_index)

    path_to_finish = matrix[end_y][end_x]
    path = []
    path.append((end_x, end_y))

    not_start = ((path_to_finish.y != start_y) or (path_to_finish.x != start_x))

    while not_start:                                        # Make sure that it doesn't go into an infinite loop
        path.append((path_to_finish.x, path_to_finish.y))
        ptfy = path_to_finish.parent_y
        ptfx = path_to_finish.parent_x
        path_to_finish = matrix[ptfy][ptfx]

    path.append((start_x, start_y))
    return path


# Driver code


# Draw the start and end circles
img = cv2.imread('test.png')
cv2.circle(img, (170,320) , 4, (255,0,0), -1) # End circle (Bottom)
cv2.circle(img, (155,5), 3, (0,0,255), -1) # Start Circle (top)


# Solve maze and print produced image
img = cv2.imread('test.png')
solved_path = solver(img, (155,5), (170,320))
draw(img, solved_path)
cv2.imwrite('test_ans.png', img)





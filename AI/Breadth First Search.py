graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F','G'],
  'D' : ['H','I'],
  'E' : ['J','K'],
  'F' : ['L'],
  'G' : ['M'],
  'H' : [],
  'I' : [],
  'J' : [],
  'K' : [],
  'L' : [],
  'M' : [],
}

visited = []
queue = []   

def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)

  while queue:
    s = queue.pop(0) 
    print (s, end = " ") 

    for neighbour in graph[s]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

bfs(visited, graph, 'A')





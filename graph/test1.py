from graphviz import Digraph

dot = Digraph(comment='这是一个有向图')
dot.node('R', 'S_ROOT')
dot.node('A', 'AWAKE')
dot.node('B', 'SLEEP')

dot.edges(['RA', 'RB'])
print(dot.source)
dot.render('output-graph.gv', view=True)